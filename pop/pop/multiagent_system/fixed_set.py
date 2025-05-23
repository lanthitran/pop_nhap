from typing import AbstractSet, Iterable, Optional

import pylru

"""
FixedSet implements a size-limited set using LRU (Least Recently Used) cache.
This class is useful for maintaining a fixed-size collection of unique items
where older items are automatically removed when capacity is reached.

Key features:
- Fixed maximum size
- LRU eviction policy
- Set-like interface
- Memory efficient for large collections
| Hung |
"""

class FixedSet(AbstractSet):
    """
    A set implementation with fixed maximum size using LRU cache.
    
    Args:
        size: Maximum number of items the set can hold
        iterable: Optional initial items to add to the set
    
    The set will automatically remove least recently used items
    when capacity is reached.
    | Hung |
    """
    def __init__(self, size: int, iterable: Optional[Iterable] = None):
        # Initialize LRU cache with specified size limit | Hung |
        self.cache: pylru.lrucache = pylru.lrucache(size=size)
        if iterable is None:
            return

        # Add all items from iterable to the set | Hung |
        for x in iterable:
            self.add(x)

    def __contains__(self, x: object) -> bool:
        """
        Check if item exists in set.
        Returns True if item is present, False otherwise.
        | Hung |
        """
        return x in self.cache

    def __len__(self) -> int:
        """
        Get current number of items in set.
        | Hung |
        """
        return len(self.cache)

    def __iter__(self):
        """
        Get iterator over set items.
        | Hung |
        """
        return self.cache.__iter__()

    def __str__(self):
        """
        Get string representation of set contents.
        | Hung |
        """
        return set(self.cache.keys()).__str__()

    def add(self, x: object):
        """
        Add item to set. If set is at capacity, least recently used
        item will be removed.
        | Hung |
        """
        self.cache[x] = None
