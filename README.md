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
      - [ ] Visualize the count distribution of the Source Name
      - [ ] Visualize the embedding of the Target Name for the source name
      - [ ] Visualize the embedding of the Event Text for the target name