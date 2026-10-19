"""Predict 5'UTR mean ribosome load from sequence, and design optimised 5'UTRs."""

__version__ = "0.1.0"

# The MPRA library is a fixed-length random 50-mer immediately upstream of the
# reporter start codon. Several modules depend on that geometry (frame of an
# upstream AUG is computed relative to the 3' end), so it lives here as the one
# source of truth.
UTR_LENGTH = 50

__all__ = ["UTR_LENGTH", "__version__"]
