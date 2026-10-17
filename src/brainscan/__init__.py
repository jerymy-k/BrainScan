"""BrainScan: brain tumour classification from MRI scans with a CNN.

Modules:
    config    paths, image size, classes and training defaults
    data      dataset cleaning, resizing, loading and splitting
    model     CNN architecture
    train     end-to-end training pipeline (``brainscan-train``)
    evaluate  test-set evaluation and reports (``brainscan-evaluate``)
    predict   single-image prediction used by the Streamlit app
"""

__version__ = '0.1.0'
