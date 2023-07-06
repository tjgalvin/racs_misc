# racs_misc

## Steps to calibrate (using Hyperdrive)

`Hyperdrive` is a replacement of the Real Time System (RTS) used at the MWA. It is written in `rust`, and can perform DI-calibration (and soon DD/Peeling). It is being used for this experiment. 

First, for this sky-model calibration stratedgy to work, a sky-model needs to be created. Make sure that the `fix_dir.py` script (written by Emil Lenc) has been appliede to the MS in question. This will ensure that the metadata describing where the individual beams are pointed towards are correct. 

Then run something like:

```
python ../ASKAP\ Sky\ Model/extract_model_for_ms.py "${MS}" --cata-dir ../ASKAP_Sky_Model/ --fwhm-scale 2 --flux-cutoff 0.02
../hyperdrive/hyperdrive-cpu di-c -d "${MS}" -s "${name}.hyp.yaml" --no-beam --ignore-weights --outputs "${SOLS}" --max-iterations 50 --stop-thresh 1e-20
singularity run /scratch2/projects/cass_glass/containers/gleamx_testing_small.img applysolutions -nocopy "${MS}" "${SOLS}"
```

The last step is a gotcha. The MS written out by `hyperdrive solutions-apply` assumes the MWA, and as such some of data in `ANTENNAS` table is not consistent with the initial MS. It seems that the `applysolutions` code is better behaved here. 
