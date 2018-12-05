

scp /Users/KevinBu/Desktop/Clemente\ Lab/imageproc/data/mixed_graphs/* buk02@mothra.hpc.mssm.edu:/sc/orga/work/buk02/clemente_lab/imageproc/data/mixed_graphs/


module load python/2.7.14
module load py_packages/2.7

mkdir /sc/orga/work/buk02/clemente_lab/imageproc/data_analysis/norm_ln_test4/

# mixed_graphs_test

python /sc/orga/work/buk02/clemente_lab/imageproc/scripts/compare.py -p /sc/orga/work/buk02/clemente_lab/imageproc/data/mixed_graphs_test/

# mixed_graphs_test_complex

python /sc/orga/work/buk02/clemente_lab/imageproc/scripts/compare.py -p /sc/orga/work/buk02/clemente_lab/imageproc/data/mixed_graphs_test_complex/

#
python /sc/orga/work/buk02/clemente_lab/imageproc/scripts/compare.py -p /sc/orga/work/buk02/clemente_lab/imageproc/data/WHO_test_biglarge/

python /sc/orga/work/buk02/clemente_lab/imageproc/scripts/compare.py -p /sc/orga/work/buk02/clemente_lab/imageproc/data/WHO_small/

python /sc/orga/work/buk02/clemente_lab/imageproc/scripts/compare.py -p /sc/orga/work/buk02/clemente_lab/imageproc/data/WHO_medium/


# imageproc



