These were taken from the ESWC21 paper Do Embeddings Actually Capture Knowledge Graph Semantics? by Jain et al.
They were obtained by modifying their code and dumping the exact same splits they use.
These particular splits are their Level-3-artist experiment for FB237.
This does not match the name from their experiments in the Table 2 from their paper.
I guess due to incorrect reporting, since this is what their code produces.
If the number of classes here does not match what Table 2 in their paper describes, it's because
    they drop clases that have less than 40 examples, but don't report this I guess.
