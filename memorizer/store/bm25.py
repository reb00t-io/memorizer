"""Local BM25 sparse encoder for Qdrant sparse vectors.

Tokenises text and emits term frequencies as sparse-vector weights; Qdrant
applies the IDF component via ``Modifier.IDF`` on the sparse index. Adapted from
the reference indexing pipeline so memorizer has no extra runtime dependency for
the lexical side of hybrid search.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

# Common stop words for DE and EN.
STOP_WORDS = frozenset(
    # English
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but if or because until while that this these those it its he she they them "
    # German
    "der die das ein eine einer eines einem einen ist sind war waren wird werden "
    "wurde wurden hat haben hatte hatten sein seine seinem seinen seiner kann "
    "können konnte konnten soll sollen sollte sollten muss müssen musste mussten "
    "darf dürfen durfte durften will wollen wollte wollten mag mögen mochte "
    "mochten und oder aber wenn weil dass ob nicht kein keine keiner keines "
    "keinem keinen auch noch schon nur sehr viel mehr als wie von zu zur zum "
    "mit für auf aus bei nach über um durch vor hinter neben zwischen an in "
    "es er sie wir ihr ich mich mir dich dir uns euch sich den dem des".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stop words / short tokens / digits."""
    tokens = re.findall(r"\b\w{2,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]


class BM25Encoder:
    """Encodes text into sparse vectors (token_id -> weight) for BM25 retrieval.

    Term frequencies are stored as weights (``log1p(tf)``); Qdrant's IDF modifier
    supplies the inverse-document-frequency component at query time.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self._next_id = 0

    def _get_token_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self._next_id += 1
        return self.vocab[token]

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """Encode a document into a sparse vector, growing the vocabulary."""
        tf = Counter(tokenize(text))
        indices: list[int] = []
        values: list[float] = []
        for token, count in tf.items():
            indices.append(self._get_token_id(token))
            values.append(math.log1p(count))
        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Encode a query using only tokens already in the vocabulary."""
        tf = Counter(tokenize(text))
        indices: list[int] = []
        values: list[float] = []
        for token, count in tf.items():
            token_id = self.vocab.get(token)
            if token_id is not None:
                indices.append(token_id)
                values.append(math.log1p(count))
        return indices, values

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.vocab), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            self.vocab = json.loads(path.read_text(encoding="utf-8"))
            self._next_id = max(self.vocab.values()) + 1 if self.vocab else 0
