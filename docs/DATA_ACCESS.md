# Source data and access

## HCP Young Adult / S1200

Obtain data through the Human Connectome Project's official access route and follow its current data-use terms:

- https://www.humanconnectome.org/study/hcp-young-adult
- https://www.humanconnectome.org/study/hcp-young-adult/document/data-use-terms

HCP source imaging, individual demographic records, derived participant-level records, credentials, and local identity maps are not bundled. Do not upload a locally obtained HCP dataset into this repository. Use authorized local storage for training inputs and preserve participant/session separation in the split manifest.

## Midnight Scan Club

MSC is available from OpenNeuro as `ds000224`:

- https://openneuro.org/datasets/ds000224
- https://doi.org/10.18112/openneuro.ds000224.v1.0.1
- Gordon et al. (2017), https://doi.org/10.1016/j.neuron.2017.07.011

Use the original release and its applicable terms rather than a redistributed copy from this repository.

## FIX training examples and descriptors

Obtain FIX and the applicable training resources from the FSL/FIX provider:

- https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/fix.html
- Salimi-Khorshidi et al. (2014), https://doi.org/10.1016/j.neuroimage.2013.11.046
- Griffanti et al. (2017), https://doi.org/10.1016/j.neuroimage.2016.12.036

The repository contains feature-alignment code, not a redistribution of the full FIX training archive. Reference examples supplied to an LLM must follow the source dataset's access conditions and the study's split policy. Synthetic format examples here contain no real FIX feature vectors.

## Included data

`data/` holds dataset-level or demographic-group aggregate statistics and manuscript-reported tables. `examples/` holds artificial examples. No individual participant's images, time series, demographics, or identity mapping is included. Model checkpoints and data-dependent manifests are not included in this snapshot.
