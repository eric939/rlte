"""Small fallback for the subset of `sortedcontainers` used by this repo."""

from __future__ import annotations

import bisect
from collections.abc import MutableMapping


class SortedList(list):
    """Minimal sorted list with insertion support."""

    def __init__(self, iterable=()):
        super().__init__(sorted(iterable))

    def add(self, value):
        bisect.insort(self, value)


class SortedDict(MutableMapping):
    """Minimal sorted dict exposing indexable `keys()` like sortedcontainers."""

    def __init__(self, key=None, *args, **kwargs):
        self._key = (lambda value: value) if key is None else key
        self._store = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._store)

    def keys(self):
        return sorted(self._store.keys(), key=self._key)

    def items(self):
        return [(key, self._store[key]) for key in self.keys()]

    def values(self):
        return [self._store[key] for key in self.keys()]

    def pop(self, key, default=None):
        if default is None:
            return self._store.pop(key)
        return self._store.pop(key, default)
