## Preprocessing

Since the timeseries data obtained from USA on the county level have certain inconsistencies, 
we tried several imputation techniques such as
- Interpolation

The US data is cumulative, so it is converted to daily counts before further preprocessing.
For the state-level analysis, we consider the latest date over associated counties for the interventions, 
and the sum across counties for calculation of infections and deaths. The FIPS code is known to be the only 
reliable and consistent identifier, so we do our analysis based on that.

Following the Imperial College analysis, the start-date for analysis is taken to be one month before
the cumulative sum of deaths exceeds 10. This keeps the windows open to account for the infective phase 
that is known to start at least 14 days earlier.

To run the data_parser in itself, uncomment lines 581-589 and call `python scripts/data_parser.py` 
from the base directory. To change the time period considered for analysis, change the value of `N2`. Currently,
it is taken to be 75 (for Europe data) and 100 (for US data).


