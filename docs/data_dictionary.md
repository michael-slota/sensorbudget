# Data dictionary

The source dataset records one office room at approximately one-minute
intervals. This dictionary should be verified by the ingestion pipeline rather
than treated as a substitute for validation.

| Variable | Role | Meaning | Unit | Modeling note |
|---|---|---|---|---|
| `source_row_id` | Identifier | Row identifier from the source file | — | Exclude from features; unique only within a source split |
| `date` | Feature | Observation timestamp | datetime | Preserve for ordering; derive calendar features cautiously |
| `Temperature` | Feature | Room air temperature | °C | Slowly varying |
| `Humidity` | Feature | Relative humidity | % | Related to humidity ratio |
| `Light` | Feature | Measured illumination | lux | Strong proxy and possible shortcut |
| `CO2` | Feature | Carbon-dioxide concentration | ppm | May respond to occupancy with a delay |
| `HumidityRatio` | Feature | Water-vapor to dry-air ratio | kg/kg | Derived from temperature and humidity |
| `Occupancy` | Target | Whether the room is occupied | 0/1 | `1` occupied; `0` unoccupied |

## Derived-feature candidates

Candidates should be added incrementally and evaluated through ablation:

- hour of day and day of week;
- elapsed time since previous observation;
- first differences for CO2, light, humidity, and temperature;
- rolling means and slopes over 5, 15, and 30 minutes;
- lagged sensor readings;
- sensor-missingness indicators.

Calendar variables can inflate apparent performance by learning the observed
office schedule. Always report results both with and without them.
