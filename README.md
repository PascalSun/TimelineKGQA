# TemporalLogicKGQA

The project will have two main parts:

- QA pair generation
- QA model training




## TODO

### Paper Thoughts

General structure:

- Problem: Current datasets for temporal question answering not classify the questions probably
    - tables to illustrate the history changes and source of the data
    - tables to show the Datasets statistics, together with the following one
- We propose a new way to classify them with reasoning ability levels and aspects
    - figures to show the categories
- What we have done
    - classify current datasets into these three categories
        - statistics table as mentioned above
    - a framework and package to generate the questions
        - design about the package
    - evaluate them with GPT-4, and state-of-the-art models
        - tables for the performance
- Our models?
- Results
    - which is the result part
    - analysis results with figures

### [Generator](./tkgqa_generator/README.md)

- [ ] ICEWS
    - [x] Load all datasets into a database, PostGIS, as it has GPS with it
    - [x] Explore further about the Sectors
    - [x] Figure out the "Source Name" and "Target Name" with high occurrence and low sector level
    - [ ] Visualize for a given source name => timeline
        - [x] Visualize the temporal logic for the source name along the timeline
        - [x] Similarity Matrix between all the SPO embeddings for the given source name along the timeline
        - [x] Visualize the embedding of the Target Name(O) for the source name (S), along the timeline, mark the line
          with color based on similarity
    - [ ] ICEWS Actor Unified KG
        - [ ] Verify the alias, make sure there are no entities are actually the same