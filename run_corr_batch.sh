for f in input_files/*.py; do
  python3 run_corr_fits.py "$f" "simult_baryons_gmo" pdf
done
