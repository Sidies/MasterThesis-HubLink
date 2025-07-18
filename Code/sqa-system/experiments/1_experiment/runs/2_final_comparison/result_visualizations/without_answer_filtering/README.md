HubLink adds a list of references at the end of its output. This is an issue when calculating some metrics like precision. Therefore we filter the output by removing the references from the output using the `BasePlotter` class.

The table in this folder is a prior version where the references were not filtered. 