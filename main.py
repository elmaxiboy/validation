from filter import (filter_single_family_detached,
                    get_simulated_years,
                    to_object_file)
from generate_typology import (calculate_rc,
                               generate_buildings,
                               generate_windows_tipology)
from validate import (derive_hvac,
                      derive_internal_gains,
                      derive_occupancy_schedule, derive_solar_gains, summarize_hvac)
from entise.constants.columns import Columns

#filter_single_family_detached()
#
#get_simulated_years()
#
#generate_buildings()
#
#calculate_rc()
#
#generate_windows_tipology()
#
#to_object_file()

#derive_occupancy_schedule()

#derive_internal_gains()

#derive_solar_gains()
derive_hvac(Columns.OCCUPANCY_GEOMA)
derive_hvac(Columns.OCCUPANCY_PHT)

summarize_hvac(Columns.OCCUPANCY_GEOMA)
summarize_hvac(Columns.OCCUPANCY_PHT)
