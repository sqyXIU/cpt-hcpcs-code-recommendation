# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
CPT / HCPCS Code Knowledge Base loader with TF-IDF similarity search.

Wraps the 2026 snapshot CSV (``codes_with_ranges.csv``) and provides:

* exact lookups: description, short description, hierarchy, system
* text similarity search over code descriptions (TF-IDF, no GPU)

.. note::
   The 2026 snapshot renamed the ``code_category`` column to
   ``code_system`` (values remain ``"CPT"`` / ``"HCPCS"``).  The
   :meth:`category` method is retained as a deprecated alias for
   :meth:`system` so existing callers keep working.

Usage::

    kb = CodeKnowledgeBase.from_csv("data/kb/codes_with_ranges.csv")
    kb.description("43235")
    # -> "Esophagogastroduodenoscopy, ..."
    kb.hierarchy("43235")
    # -> [("43200-43273", "Esophagoscopy, ..."), ...]
    results = kb.search("EGD with biopsy", top_k=10)
    # -> [("43239", 0.42), ("43235", 0.39), ...]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_RANGE_LEVELS = [1, 2, 3, 4, 5, 6]


class CodeKnowledgeBase:
    """In-memory CPT / HCPCS knowledge base with optional TF-IDF index."""

    def __init__(self, df: pd.DataFrame) -> None:
        # Normalize the code column (always uppercase, stripped)
        df = df.copy()
        df["code"] = df["code"].astype(str).str.strip().str.upper()
        df = df.drop_duplicates(subset=["code"], keep="first")
        df = df.set_index("code")
        self._df = df

        # Fast set for membership tests
        self._code_set: Set[str] = set(df.index)

        # Lazy-initialized TF-IDF artifacts
        self._tfidf_matrix = None
        self._tfidf_vectorizer = None
        self._tfidf_codes: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        csv_path: Path | str,
        build_index: bool = True,
    ) -> "CodeKnowledgeBase":
        """
        Load from the codes_with_ranges.csv file.

        Parameters
        ----------
        csv_path : path to CSV
        build_index : if True, immediately build the TF-IDF index
        """
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        LOGGER.info("Loaded %d codes from %s", len(df), Path(csv_path).name)
        kb = cls(df)
        if build_index:
            kb.build_tfidf_index()
        return kb

    # ------------------------------------------------------------------
    # Exact lookups
    # ------------------------------------------------------------------
    @property
    def codes(self) -> Set[str]:
        """Set of all known code strings."""
        return self._code_set

    def __contains__(self, code: str) -> bool:
        return code.strip().upper() in self._code_set

    def __len__(self) -> int:
        return len(self._code_set)

    def _get(self, code: str, col: str) -> Optional[str]:
        code = code.strip().upper()
        if code not in self._code_set:
            return None
        val = self._df.at[code, col]
        if pd.isna(val):
            return None
        return str(val)

    def description(self, code: str) -> Optional[str]:
        """Full code description, or None if code not found."""
        return self._get(code, "code_description")

    def short_description(self, code: str) -> Optional[str]:
        """
        First sentence of the description (suitable for prompts), or None.
        """
        full = self.description(code)
        if full is None:
            return None
        # Take up to the first period followed by whitespace or end
        m = re.match(r"^(.+?\.)\s", full)
        return m.group(1) if m else full[:200]

    def lay_term(self, code: str) -> Optional[str]:
        """Lay-term description, or None."""
        return self._get(code, "code_lay_term")

    def system(self, code: str) -> Optional[str]:
        """``'CPT'`` or ``'HCPCS'``, or None.

        Reads the ``code_system`` column introduced in the 2026 KB
        snapshot (renamed from ``code_category`` in 2025 and earlier).
        """
        return self._get(code, "code_system")

    def category(self, code: str) -> Optional[str]:
        """Deprecated alias for :meth:`system` (2025-era column name).

        Kept so callers written against the 2025 snapshot keep working.
        Prefer :meth:`system` in new code.
        """
        return self.system(code)

    def hierarchy(self, code: str) -> List[Tuple[str, str]]:
        """
        Return the code-range hierarchy as a list of
        ``(range_code, range_description)`` tuples, from broadest to narrowest.

        Returns an empty list if the code is unknown.
        """
        code = code.strip().upper()
        if code not in self._code_set:
            return []
        row = self._df.loc[code]
        levels = []
        for lvl in _RANGE_LEVELS:
            rng = row.get(f"code_range_{lvl}")
            desc = row.get(f"code_range_{lvl}_description")
            if pd.isna(rng) or not str(rng).strip():
                break
            levels.append((str(rng).strip(), str(desc).strip() if pd.notna(desc) else ""))
        return levels

    def family(self, code: str) -> Optional[str]:
        """
        Return the most specific (deepest) range code for grouping, or None.

        Useful for same-family confounders in negative sampling.
        """
        h = self.hierarchy(code)
        return h[-1][0] if h else None

    def family_codes(self, code: str) -> Set[str]:
        """
        Return all codes sharing the same deepest range as *code*.
        """
        fam = self.family(code)
        if fam is None:
            return set()
        # Find which range level this family corresponds to
        code_up = code.strip().upper()
        h = self.hierarchy(code_up)
        if not h:
            return set()
        level = len(h)  # deepest level
        col = f"code_range_{level}"
        mask = self._df[col].astype(str).str.strip() == fam
        return set(self._df.index[mask])

    # ------------------------------------------------------------------
    # TF-IDF search
    # ------------------------------------------------------------------
    def build_tfidf_index(self) -> None:
        """Build (or rebuild) the TF-IDF index over code descriptions."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        codes = list(self._df.index)
        # Combine code + short description for better lexical matching
        texts = []
        for code in codes:
            desc = self._df.at[code, "code_description"]
            desc = str(desc) if pd.notna(desc) else ""
            texts.append(f"{code} {desc}")

        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        self._tfidf_codes = codes
        LOGGER.info("TF-IDF index built: %d codes, %d features",
                     len(codes), self._tfidf_matrix.shape[1])

    def search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the *top_k* most similar codes to *query_text* by TF-IDF
        cosine similarity.

        Returns a list of ``(code, score)`` tuples, sorted descending.
        Requires :meth:`build_tfidf_index` to have been called.
        """
        if self._tfidf_matrix is None or self._tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF index not built. Call build_tfidf_index() first.")

        query_vec = self._tfidf_vectorizer.transform([query_text])
        scores = (self._tfidf_matrix @ query_vec.T).toarray().ravel()

        # Partial sort for efficiency
        if top_k < len(scores):
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)

        return [(self._tfidf_codes[i], float(scores[i])) for i in top_idx[:top_k]]

    def batch_search(
        self,
        query_texts: List[str],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Vectorized batch search for multiple queries.
        """
        if self._tfidf_matrix is None or self._tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF index not built.")

        query_mat = self._tfidf_vectorizer.transform(query_texts)
        score_mat = (self._tfidf_matrix @ query_mat.T).toarray()  # (n_codes, n_queries)

        results = []
        for col in range(score_mat.shape[1]):
            scores = score_mat[:, col]
            if top_k < len(scores):
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
            else:
                top_idx = np.argsort(-scores)
            results.append(
                [(self._tfidf_codes[i], float(scores[i])) for i in top_idx[:top_k]]
            )
        return results
