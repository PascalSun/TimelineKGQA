# TemporalLogicKGQA

The project will have two main parts:

- QA pair generation
- QA model training

## TODO

### [Generator](./tkgqa_generator/README.md)

- [ ] ICEWS
    - [x] Load all datasets into a database, PostGIS, as it has GPS with it
    - [x] Explore further about the Sectors
    - [x] Figure out the "Source Name" and "Target Name" with high occurrence and low sector level
    - [ ] Visualize for a given source name => timeline
        - [x] Visualize the temporal logic for the source name along the timeline
        - [x] Similarity Matrix between all the SPO embeddings for the given source name along the timeline
        - [ ] Visualize the embedding of the Target Name(O) for the source name (S), along the timeline, mark the line
          with color based on similarity
    - [ ] ICEWS Actor Unified KG
        - [ ] Verify the alias, make sure there are no entities are actually the same