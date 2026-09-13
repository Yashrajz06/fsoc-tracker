"""Optional CNN candidate discriminator.

The classical path stays primary and always localises. Nothing in this package estimates a
position: the network scores candidate blobs the classical detector has already found, and is
consulted only where flux ranking provably fails. See :mod:`src.ai.validator`.
"""
