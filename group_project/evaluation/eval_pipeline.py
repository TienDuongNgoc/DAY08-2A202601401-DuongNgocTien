"""Evaluate the university-services RAG pipeline and export ``results.md``.

Two execution modes are provided:

``offline`` (default)
    A deterministic, dependency-free benchmark.  It uses the same standardized
    corpus, compares hybrid retrieval + reranking with dense-only retrieval,
    and calculates transparent lexical proxy metrics.  This mode is useful in
    CI and before spending LLM/API quota.

``ragas``
    Runs the project retrieval components, generates answers with an LLM, and
    evaluates them with RAGAS using faithfulness, answer relevancy, context
    recall, and context precision.

Examples::

    python -m group_project.evaluation.eval_pipeline
    python -m group_project.evaluation.eval_pipeline --mode ragas --limit 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from dotenv import load_dotenv


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
SRC_DIR = PROJECT_ROOT / "src"

METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
)

STOP_WORDS = {
    "a", "ai", "anh", "bao", "bằng", "bị", "các", "cái", "cho", "chương",
    "có", "của", "do", "dành", "giữa", "hay", "học", "khi", "không", "là",
    "làm", "một", "nào", "những", "này", "phải", "sinh", "sẽ", "sau", "tại",
    "theo", "thì", "thuộc", "trong", "trên", "trước", "từ", "và", "vào", "với",
    "được", "để", "đến", "đối", "ở", "the", "and", "for", "from", "in", "of",
    "on", "or", "to", "with",
}


@dataclass(frozen=True)
class EvalConfig:
    """A retrieval configuration included in the A/B benchmark."""

    name: str
    label: str
    description: str
    strategy: str


CONFIGS = (
    EvalConfig(
        name="hybrid_rerank",
        label="Config A (hybrid + rerank)",
        description=(
            "Kết hợp dense retrieval và BM25 bằng Reciprocal Rank Fusion, sau đó "
            "rerank theo độ tương đồng và mức bao phủ từ khóa."
        ),
        strategy="hybrid",
    ),
    EvalConfig(
        name="dense_only",
        label="Config B (dense-only)",
        description="Chỉ sử dụng dense retrieval, không BM25 và không reranking.",
        strategy="dense",
    ),
)


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict[str, str]]:
    """Load and validate the golden dataset."""

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or len(dataset) < 15:
        raise ValueError("Golden dataset phải là một danh sách có ít nhất 15 mẫu.")

    required = {"question", "expected_answer", "expected_context"}
    seen_questions: set[str] = set()
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Mẫu #{index} thiếu một trong các trường: {sorted(required)}")
        for key in required:
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"Mẫu #{index} có trường {key!r} rỗng hoặc không phải chuỗi.")
        if item["question"] in seen_questions:
            raise ValueError(f"Câu hỏi trùng lặp: {item['question']}")
        seen_questions.add(item["question"])

    return dataset


def _strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def _tokens(text: str, *, remove_stop_words: bool = False) -> list[str]:
    normalized = _strip_accents(unicodedata.normalize("NFC", text)).lower()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if remove_stop_words:
        normalized_stops = {_strip_accents(word).lower() for word in STOP_WORDS}
        tokens = [token for token in tokens if token not in normalized_stops and len(token) > 1]
    return tokens


def _semantic_terms(text: str) -> list[str]:
    """Return unigram and bigram features for the offline dense proxy."""

    words = _tokens(text, remove_stop_words=True)
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _f1(left: Iterable[str], right: Iterable[str]) -> float:
    left_counter, right_counter = Counter(left), Counter(right)
    overlap = sum((left_counter & right_counter).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(left_counter.values())
    recall = overlap / sum(right_counter.values())
    return 2 * precision * recall / (precision + recall)


def _recall(reference: Iterable[str], candidate: Iterable[str]) -> float:
    reference_counter, candidate_counter = Counter(reference), Counter(candidate)
    denominator = sum(reference_counter.values())
    return sum((reference_counter & candidate_counter).values()) / denominator if denominator else 0.0


def _precision(candidate: Iterable[str], evidence: Iterable[str]) -> float:
    candidate_counter, evidence_counter = Counter(candidate), Counter(evidence)
    denominator = sum(candidate_counter.values())
    return sum((candidate_counter & evidence_counter).values()) / denominator if denominator else 0.0


def _chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 120) -> list[str]:
    """Split Markdown into paragraph-aware chunks without third-party packages."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            step = max_chars - overlap_chars
            for start in range(0, len(paragraph), step):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap_chars:].strip()
            current = f"{tail}\n\n{paragraph}".strip()

    if current:
        chunks.append(current)
    return chunks


def _prepare_markdown(text: str) -> str:
    """Remove crawl navigation/URL noise while preserving article and table text."""

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        # Crawled RMIT pages contain a very large navigation menu made almost
        # entirely of linked bullet items.  Keeping it causes tuition queries to
        # retrieve a course page merely because its menu links to "Kinh doanh".
        if re.match(r"^\s*[*+-]\s+\[[^]]*]\(https?://", line):
            continue
        line = re.sub(r"!\[[^]]*]\([^)]*\)", "", line)
        line = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", line)
        line = re.sub(r"https?://\S+", " ", line)
        if "javascript:void" in line.lower() or "skip to content" in line.lower():
            continue
        if line.strip():
            cleaned_lines.append(line.rstrip())
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")
    return "\n".join(cleaned_lines).strip()


class OfflineIndex:
    """Dependency-free dense/BM25 index used by the reproducible offline mode."""

    def __init__(self, corpus_dir: Path = STANDARDIZED_DIR) -> None:
        self.documents: list[dict[str, Any]] = []
        for path in sorted(corpus_dir.rglob("*.md")):
            content = _prepare_markdown(path.read_text(encoding="utf-8"))
            for chunk_index, chunk in enumerate(_chunk_text(content)):
                self.documents.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "source": path.name,
                            "type": "legal" if "legal" in path.parts else "news",
                            "path": str(path.relative_to(corpus_dir)).replace("\\", "/"),
                            "chunk_index": chunk_index,
                        },
                    }
                )

        if not self.documents:
            raise ValueError(f"Không tìm thấy tài liệu Markdown trong {corpus_dir}")

        self.word_tokens = [_tokens(item["content"], remove_stop_words=True) for item in self.documents]
        self.semantic_tokens = [_semantic_terms(item["content"]) for item in self.documents]
        self.average_length = fmean(len(tokens) for tokens in self.word_tokens)

        self.word_df = Counter(
            term for tokens in self.word_tokens for term in set(tokens)
        )
        self.semantic_df = Counter(
            term for tokens in self.semantic_tokens for term in set(tokens)
        )
        self.semantic_vectors = [self._tfidf_vector(tokens) for tokens in self.semantic_tokens]

    def _idf(self, term: str, *, semantic: bool = False) -> float:
        document_frequency = (self.semantic_df if semantic else self.word_df).get(term, 0)
        return math.log((len(self.documents) + 1) / (document_frequency + 1)) + 1.0

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {
            term: (1.0 + math.log(count)) * self._idf(term, semantic=True)
            for term, count in counts.items()
        }

    def dense_scores(self, query: str) -> list[float]:
        query_vector = self._tfidf_vector(_semantic_terms(query))
        return [_cosine(query_vector, vector) for vector in self.semantic_vectors]

    def bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_terms = set(_tokens(query, remove_stop_words=True))
        scores: list[float] = []
        for tokens in self.word_tokens:
            counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.word_df.get(term, 0)
                idf = math.log(
                    1.0 + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (1.0 - b + b * length / self.average_length)
                score += idf * (frequency * (k1 + 1.0)) / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        maximum = max(scores, default=0.0)
        return [score / maximum if maximum else 0.0 for score in scores]

    def search(self, query: str, strategy: str, top_k: int = 5) -> list[dict[str, Any]]:
        dense = self.dense_scores(query)
        if strategy == "dense":
            ranked = sorted(range(len(dense)), key=dense.__getitem__, reverse=True)[:top_k]
            return [
                {**self.documents[index], "score": dense[index], "source": "dense"}
                for index in ranked
                if dense[index] > 0
            ]

        sparse = self.bm25_scores(query)
        dense_rank = sorted(range(len(dense)), key=dense.__getitem__, reverse=True)[: top_k * 4]
        sparse_rank = sorted(range(len(sparse)), key=sparse.__getitem__, reverse=True)[: top_k * 4]

        rrf_scores: Counter[int] = Counter()
        for ranked_list in (dense_rank, sparse_rank):
            for rank, index in enumerate(ranked_list, start=1):
                rrf_scores[index] += 1.0 / (60 + rank)

        dense_normalized = self._normalize(dense)
        sparse_normalized = self._normalize(sparse)
        query_terms = set(_tokens(query, remove_stop_words=True))
        reranked: list[tuple[int, float]] = []
        for index in rrf_scores:
            document_terms = set(self.word_tokens[index])
            coverage = len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0
            score = (
                0.25 * rrf_scores[index] / max(rrf_scores.values())
                + 0.35 * dense_normalized[index]
                + 0.30 * sparse_normalized[index]
                + 0.10 * coverage
            )
            reranked.append((index, score))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return [
            {**self.documents[index], "score": score, "source": "hybrid"}
            for index, score in reranked[:top_k]
        ]


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^]]*]\([^)]*\)", "", text)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*_`|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extractive_answer(question: str, contexts: list[str], max_sentences: int = 3) -> str:
    """Create a non-LLM answer from the most query-relevant evidence sentences."""

    query_terms = set(_tokens(question, remove_stop_words=True))
    candidates: list[tuple[float, str]] = []
    for context_rank, context in enumerate(contexts):
        sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
        for sentence in sentences:
            cleaned = _clean_markdown(sentence)
            if not 25 <= len(cleaned) <= 500:
                continue
            sentence_terms = set(_tokens(cleaned, remove_stop_words=True))
            if not sentence_terms:
                continue
            overlap = len(query_terms & sentence_terms)
            score = overlap / max(len(query_terms), 1) + 0.05 / (context_rank + 1)
            if any(character.isdigit() for character in cleaned):
                score += 0.03
            candidates.append((score, cleaned))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    for score, sentence in candidates:
        if score <= 0.05:
            continue
        sentence_tokens = _tokens(sentence, remove_stop_words=True)
        if any(_f1(sentence_tokens, _tokens(existing, remove_stop_words=True)) > 0.8 for existing in selected):
            continue
        selected.append(sentence)
        if len(selected) == max_sentences:
            break

    return " ".join(selected) if selected else "Không đủ thông tin trong ngữ cảnh được truy xuất."


def _offline_metrics(
    question: str,
    answer: str,
    expected_answer: str,
    contexts: list[str],
) -> dict[str, float]:
    question_tokens = _tokens(question, remove_stop_words=True)
    answer_tokens = _tokens(answer, remove_stop_words=True)
    expected_tokens = _tokens(expected_answer, remove_stop_words=True)
    context_tokens = _tokens(" ".join(contexts), remove_stop_words=True)

    faithfulness = _precision(answer_tokens, context_tokens)
    # In offline mode, answer correctness F1 is the transparent proxy for RAGAS answer relevancy.
    answer_relevance = 0.8 * _f1(answer_tokens, expected_tokens) + 0.2 * _recall(question_tokens, answer_tokens)
    context_recall = _recall(expected_tokens, context_tokens)

    graded_relevance: list[float] = []
    for context in contexts:
        context_terms = _tokens(context, remove_stop_words=True)
        graded_relevance.append(
            min(
                1.0,
                0.7 * _recall(expected_tokens, context_terms)
                + 0.3 * _recall(question_tokens, context_terms),
            )
        )
    rank_weights = [1.0 / rank for rank in range(1, len(graded_relevance) + 1)]
    context_precision = (
        sum(score * weight for score, weight in zip(graded_relevance, rank_weights))
        / sum(rank_weights)
        if rank_weights
        else 0.0
    )

    return {
        "faithfulness": round(min(1.0, faithfulness), 4),
        "answer_relevance": round(min(1.0, answer_relevance), 4),
        "context_recall": round(min(1.0, context_recall), 4),
        "context_precision": round(min(1.0, context_precision), 4),
    }


def evaluate_offline_config(
    index: OfflineIndex,
    golden_dataset: list[dict[str, str]],
    config: EvalConfig,
    top_k: int,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for item in golden_dataset:
        retrieved = index.search(item["question"], strategy=config.strategy, top_k=top_k)
        contexts = [result["content"] for result in retrieved]
        answer = _extractive_answer(item["question"], contexts)
        scores = _offline_metrics(item["question"], answer, item["expected_answer"], contexts)
        samples.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "answer": answer,
                "contexts": contexts,
                "sources": [result["metadata"].get("source", "unknown") for result in retrieved],
                "scores": scores,
            }
        )

    overall = {
        metric: round(fmean(sample["scores"][metric] for sample in samples), 4)
        for metric in METRICS
    }
    return {"overall": overall, "samples": samples}


def _llm_settings() -> tuple[str, str, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openrouter_key and "..." not in openrouter_key and len(openrouter_key) > 20:
        return (
            openrouter_key,
            "https://openrouter.ai/api/v1",
            os.getenv("EVAL_LLM_MODEL", "inclusionai/ling-3.0-flash:free"),
        )
    if openai_key and "..." not in openai_key and len(openai_key) > 20:
        return openai_key, "https://api.openai.com/v1", os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
    raise RuntimeError("Chế độ RAGAS cần OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong file .env.")


def validate_ragas_environment() -> None:
    """Fail early with an actionable message instead of a nested import error."""

    required_modules = {
        "chromadb": "chromadb",
        "datasets": "datasets",
        "langchain_openai": "langchain-openai",
        "langchain_text_splitters": "langchain-text-splitters",
        "openai": "openai",
        "ragas": "ragas",
        "rank_bm25": "rank-bm25",
        "sentence_transformers": "sentence-transformers",
    }
    missing = [
        package
        for module, package in required_modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Chế độ RAGAS còn thiếu dependency: "
            + ", ".join(missing)
            + ". Hãy chạy: pip install -r requirements.txt"
        )
    _llm_settings()


def _generate_with_context(question: str, contexts: list[str], sources: list[str]) -> str:
    from openai import OpenAI

    api_key, base_url, model = _llm_settings()
    evidence = "\n\n".join(
        f"[Nguồn {index}: {source}]\n{context}"
        for index, (source, context) in enumerate(zip(sources, contexts), start=1)
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý RAG về chính sách và dịch vụ RMIT Việt Nam. Chỉ trả lời từ "
                    "ngữ cảnh, trả lời bằng tiếng Việt, ngắn gọn và trích nguồn dạng [Nguồn n]. "
                    "Nếu thiếu bằng chứng, hãy nói rõ không thể xác minh."
                ),
            },
            {"role": "user", "content": f"Ngữ cảnh:\n{evidence}\n\nCâu hỏi: {question}"},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def run_project_config(
    golden_dataset: list[dict[str, str]], config: EvalConfig, top_k: int
) -> list[dict[str, Any]]:
    """Run a real project retrieval configuration and generate answers for RAGAS."""

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    if config.strategy == "hybrid":
        from task9_retrieval_pipeline import retrieve

        search = lambda query: retrieve(query, top_k=top_k, use_reranking=True)
    else:
        from task5_semantic_search import semantic_search

        search = lambda query: semantic_search(query, top_k=top_k)

    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(golden_dataset, start=1):
        print(f"  [{config.name}] {index}/{len(golden_dataset)}: {item['question'][:65]}")
        retrieved = search(item["question"])
        contexts = [result["content"] for result in retrieved]
        sources = [result.get("metadata", {}).get("source", "unknown") for result in retrieved]
        answer = _generate_with_context(item["question"], contexts, sources)
        predictions.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "answer": answer,
                "contexts": contexts,
                "sources": sources,
            }
        )
    return predictions


def evaluate_with_ragas(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate prepared RAG predictions with RAGAS 0.1.x."""

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as error:
        raise RuntimeError(
            "Thiếu dependency RAGAS. Hãy chạy: pip install -r requirements.txt"
        ) from error

    api_key, base_url, model = _llm_settings()
    embedding_default = "openai/text-embedding-3-small" if "openrouter" in base_url else "text-embedding-3-small"
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", embedding_default)

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0.0)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key, base_url=base_url)
    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in predictions],
            "answer": [row["answer"] for row in predictions],
            "contexts": [row["contexts"] for row in predictions],
            "ground_truth": [row["expected_answer"] for row in predictions],
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    column_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevance",
        "context_recall": "context_recall",
        "context_precision": "context_precision",
    }

    samples: list[dict[str, Any]] = []
    for row_index, prediction in enumerate(predictions):
        scores = {
            target: round(float(frame.iloc[row_index][source]), 4)
            if not math.isnan(float(frame.iloc[row_index][source]))
            else 0.0
            for source, target in column_map.items()
        }
        samples.append({**prediction, "scores": scores})

    overall = {
        metric: round(fmean(sample["scores"][metric] for sample in samples), 4)
        for metric in METRICS
    }
    return {"overall": overall, "samples": samples}


def compare_configs(
    golden_dataset: list[dict[str, str]],
    *,
    mode: str = "offline",
    top_k: int = 5,
) -> dict[str, Any]:
    """Run and evaluate both A/B configurations."""

    comparison: dict[str, Any] = {
        "mode": mode,
        "top_k": top_k,
        "dataset_size": len(golden_dataset),
        "configs": {},
    }
    offline_index = OfflineIndex() if mode == "offline" else None

    for config in CONFIGS:
        print(f"Evaluating {config.label} ...")
        if mode == "offline":
            evaluated = evaluate_offline_config(offline_index, golden_dataset, config, top_k)  # type: ignore[arg-type]
        else:
            predictions = run_project_config(golden_dataset, config, top_k)
            evaluated = evaluate_with_ragas(predictions)
        comparison["configs"][config.name] = {
            "label": config.label,
            "description": config.description,
            **evaluated,
        }
    return comparison


def _metric_average(metrics: dict[str, float]) -> float:
    return fmean(metrics[metric] for metric in METRICS)


def _escape_table(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace("|", "\\|").strip()


def _recommendations(config_result: dict[str, Any]) -> list[tuple[str, str]]:
    scores = config_result["overall"]
    recommendations: list[tuple[str, str]] = []
    if scores["context_recall"] < 0.75:
        recommendations.append(
            (
                "Tăng context recall",
                "Hiệu chỉnh chunk size/overlap, tăng candidate pool trước rerank và bổ sung query expansion cho các câu hỏi chính sách dài.",
            )
        )
    if scores["context_precision"] < 0.75:
        recommendations.append(
            (
                "Tăng context precision",
                "Dùng cross-encoder đa ngôn ngữ thay cho lượt RRF thứ hai và lọc các đoạn menu/điều hướng trước khi lập chỉ mục.",
            )
        )
    if scores["answer_relevance"] < 0.75 or scores["faithfulness"] < 0.75:
        recommendations.append(
            (
                "Cải thiện câu trả lời",
                "Giữ temperature=0, yêu cầu câu trả lời ngắn theo từng ý có citation và từ chối trả lời khi không có bằng chứng.",
            )
        )
    recommendations.append(
        (
            "Làm sạch dữ liệu crawl",
            "Loại menu điều hướng, URL lặp và nội dung chân trang trước khi chunk/index để truy vấn không bị khớp nhầm với tên chương trình trong navigation.",
        )
    )
    recommendations.append(
        (
            "Vận hành evaluation ổn định",
            "Lưu cache câu trả lời/context, chạy thử --limit 5 trước rồi mới chạy đủ 20 mẫu để tránh lãng phí quota RAGAS.",
        )
    )
    return recommendations[:3]


def export_results(comparison: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    """Export overall scores, A/B analysis, bottom samples, and recommendations."""

    config_a = comparison["configs"]["hybrid_rerank"]
    config_b = comparison["configs"]["dense_only"]
    mode = comparison["mode"]
    scoring_method = (
        "LLM-as-a-judge"
        if mode == "ragas"
        else "Deterministic lexical scoring"
    )

    display_names = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }
    lines = [
        "# RAG Evaluation Results",
        "",
        f"- **Ngày chạy:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"- **Phương pháp chấm:** {scoring_method}",
        f"- **Golden dataset:** {comparison['dataset_size']} câu hỏi",
        f"- **Số context mỗi câu:** top_k={comparison['top_k']}",
        "",
    ]
    if mode == "offline":
        lines.extend(
            [
                "Các chỉ số được tính minh bạch như sau: Faithfulness là tỉ lệ token của câu trả lời có trong context; "
                "Answer Relevance là F1 với đáp án chuẩn kết hợp độ bao phủ câu hỏi; Context Recall là độ bao phủ "
                "đáp án chuẩn trong toàn bộ context; Context Precision là độ liên quan có trọng số theo thứ hạng.",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall Scores",
            "",
            "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A − B) |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        left = config_a["overall"][metric]
        right = config_b["overall"][metric]
        lines.append(f"| {display_names[metric]} | {left:.3f} | {right:.3f} | {left - right:+.3f} |")
    average_a = _metric_average(config_a["overall"])
    average_b = _metric_average(config_b["overall"])
    lines.extend(
        [
            f"| **Average** | **{average_a:.3f}** | **{average_b:.3f}** | **{average_a - average_b:+.3f}** |",
            "",
            "## A/B Comparison Analysis",
            "",
            f"**Config A:** {config_a['description']}",
            "",
            f"**Config B:** {config_b['description']}",
            "",
        ]
    )
    winner = "Config A" if average_a >= average_b else "Config B"
    lines.append(
        f"**Kết luận:** {winner} đạt điểm trung bình cao hơn ({max(average_a, average_b):.3f} so với "
        f"{min(average_a, average_b):.3f}). Chênh lệch lớn nhất nằm ở "
        f"{display_names[max(METRICS, key=lambda metric: abs(config_a['overall'][metric] - config_b['overall'][metric]))]}."
    )

    samples = sorted(
        config_a["samples"], key=lambda sample: _metric_average(sample["scores"])
    )[:3]
    lines.extend(
        [
            "",
            "## Worst Performers (Bottom 3 của Config A)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |",
            "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for rank, sample in enumerate(samples, start=1):
        scores = sample["scores"]
        weakest = min(METRICS, key=scores.__getitem__)
        if weakest in {"context_recall", "context_precision"}:
            stage = "Retrieval"
            cause = "Context liên quan chưa được xếp đủ cao hoặc chunk chứa nhiều nội dung nhiễu."
        else:
            stage = "Generation"
            cause = "Câu trả lời chưa bao phủ đủ ý của đáp án chuẩn hoặc thiếu diễn đạt trực tiếp."
        lines.append(
            f"| {rank} | {_escape_table(sample['question'])} | {scores['faithfulness']:.3f} | "
            f"{scores['answer_relevance']:.3f} | {scores['context_recall']:.3f} | "
            f"{scores['context_precision']:.3f} | {stage} | {cause} |"
        )

    lines.extend(["", "## Recommendations", ""])
    for index, (title, action) in enumerate(_recommendations(config_a), start=1):
        lines.extend([f"### {index}. {title}", "", f"**Action:** {action}", ""])

    lines.extend(
        [
            "## Reproduction",
            "",
            "```powershell",
            "python -m group_project.evaluation.eval_pipeline",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline and write results.md")
    parser.add_argument("--mode", choices=("offline", "ragas"), default="offline")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N câu đầu; 0 nghĩa là toàn bộ")
    parser.add_argument("--top-k", type=int, default=5, help="Số context truy xuất cho mỗi câu")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit phải lớn hơn hoặc bằng 0")
    if args.top_k <= 0:
        raise ValueError("--top-k phải lớn hơn 0")

    golden_dataset = load_golden_dataset()
    if args.limit:
        golden_dataset = golden_dataset[: args.limit]
    print(f"Loaded {len(golden_dataset)} test cases")

    if args.mode == "ragas":
        validate_ragas_environment()
    comparison = compare_configs(golden_dataset, mode=args.mode, top_k=args.top_k)
    export_results(comparison, args.output)
    print(f"Exported evaluation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
