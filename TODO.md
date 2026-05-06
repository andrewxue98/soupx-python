# TODO

- Fix `calculateContaminationFraction` so zero-observation marker sets do not crash the Poisson GLM fit and instead produce a valid near-zero contamination estimate.
- Relax gene-set validation in `estimateNonExpressingCells` so partially missing marker panels continue with genes present in the matrix rather than failing the entire panel.
- Preserve cell-ID alignment for labeled `useToEst` inputs in `calculateContaminationFraction` by realigning `Series`/`DataFrame` masks to `sc.metaData.index` before applying them.
