# Pipeline measurements

Median of three fresh-process medians; whiskers span these three medians, not a confidence interval.

Worst-rank synchronized host wall time for complete dispatch, BF16 expert MLP and combine; control barriers excluded.

Each rank r receives N + 3r input tokens. Top-k is two, with four experts per rank and an expert intermediate width of 128.

<img src="../assets/tables/measurements-2gpu.png" alt="2-GPU measurements" width="940">

<img src="../assets/tables/measurements-4gpu.png" alt="4-GPU measurements" width="940">

<img src="../assets/tables/measurements-8gpu.png" alt="8-GPU measurements" width="940">

Packed reduction compares the packed-input/registered-output path with the compact native adapter in the same process launches.

Native transport permits NVLink in these runs. RDMA-only transport is measured in a separate run.
