# Overnight tariff-bank comparison

All controllers were selected on training settlement and screened on independent validation weather. Test weather is used only for the reported held-out means.

## Selected controllers

| Tariff | Family controller | Base controller |
|---|---|---|
| fair_leg | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| fair_leg_passthrough | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| stress_export_default | 23: instant_10 | 2: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 1000.0}` |
| stress_export_tuned | 10: cap_8 | 6: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 8.0}` |
| stress_export_biting | 12: cap_2 | 15: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 2.0}` |
| stress_bidirectional | 23: instant_10 | 2: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 1000.0}` |
| stress_tenant_exempt | 23: instant_10 | 2: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 1000.0}` |
| stress_smoothed | 3: delay_11 | 2: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 1000.0}` |
| tou_midday_export | 5: stagger_11_1h | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| tou_evening_import | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| kva_coincident_charge | 31: instant_slow_stagger_1h | 2: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 1000.0}` |
| kva_charge_class_neutral | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| kva_peak_ratchet | 31: instant_slow_stagger_1h | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| kw_peak_ratchet | 31: instant_slow_stagger_1h | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| loss_share_linear | 31: instant_slow_stagger_1h | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| loss_share_quadratic | 31: instant_slow_stagger_1h | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| voltage_level_exposure | 31: instant_slow_stagger_1h | 2: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 1000.0}` |
| voltage_sensitivity | 2: delay_09 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| staggered_class_tou | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| own_peak_ratchet | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| own_deviation_charge | 31: instant_slow_stagger_1h | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| reverse_flow_charge | 31: instant_slow_stagger_1h | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| two_part_volumetric | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| two_part_fixed | 7: slow_stagger | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| tenant_floor | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| stress_funds_tenant_floor | 11: cap_4 | 10: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 4.0}` |
| stress_class_neutral | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |
| joint_system_cost | 23: instant_10 | 2: self_consumption `{"charge_after_hour": 11.0, "export_cap_kw": 1000.0}` |
| subsidy_ablation | 4: delay_13 | 3: self_consumption `{"charge_after_hour": 13.0, "export_cap_kw": 1000.0}` |

## Declared validation screens

Runner finalist: **kva_peak_ratchet**. Accepted: **true**.

| Tariff | Screen result vs tuned fair LEG | Revenue adequate |
|---|---:|---:|
| fair_leg | reference | true |
| fair_leg_passthrough | pass | true |
| stress_export_default | fail | true |
| stress_export_tuned | fail | true |
| stress_export_biting | fail | true |
| stress_bidirectional | fail | true |
| stress_tenant_exempt | fail | true |
| stress_smoothed | fail | true |
| tou_midday_export | fail | true |
| tou_evening_import | pass | true |
| kva_coincident_charge | fail | true |
| kva_charge_class_neutral | pass | true |
| kva_peak_ratchet | pass | true |
| kw_peak_ratchet | pass | true |
| loss_share_linear | fail | true |
| loss_share_quadratic | fail | true |
| voltage_level_exposure | fail | true |
| voltage_sensitivity | fail | true |
| staggered_class_tou | fail | true |
| own_peak_ratchet | pass | true |
| own_deviation_charge | fail | true |
| reverse_flow_charge | fail | true |
| two_part_volumetric | fail | true |
| two_part_fixed | fail | true |
| tenant_floor | pass | true |
| stress_funds_tenant_floor | fail | true |
| stress_class_neutral | fail | true |
| joint_system_cost | fail | true |
| subsidy_ablation | fail | false |

## Held-out test-week means

Group prices are CHF per actual load kWh. Parentheses show the difference from `fair_leg/tuned_family`; positive is worse for that group.

| Tariff | Cell role | Run | Export peak kW | Import peak kW | Max ramp kW | Curtailment | Community settlement CHF | Tenant price | PV-only price | PV+battery price | Large-flex price |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fair_leg | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| fair_leg | fixed_base | `fair_leg/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| fair_leg | tuned_base | `fair_leg/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2228 (-0.0006) | -0.2608 (-0.0011) | -0.4304 (+0.0020) | -0.1531 (+0.0020) |
| fair_leg | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| fair_leg | tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| fair_leg | voltage_family | `fair_leg/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2235 (-0.0000) | -0.2596 (+0.0001) | -0.4312 (+0.0012) | -0.1554 (-0.0003) |
| fair_leg | voltage_off_matched | `fair_leg/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2235 (+0.0000) | -0.2595 (+0.0001) | -0.4323 (+0.0001) | -0.1542 (+0.0009) |
| fair_leg_passthrough | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| fair_leg_passthrough | fixed_base | `fair_leg_passthrough/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| fair_leg_passthrough | tuned_base | `fair_leg_passthrough/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2228 (-0.0006) | -0.2608 (-0.0011) | -0.4304 (+0.0020) | -0.1531 (+0.0020) |
| fair_leg_passthrough | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| fair_leg_passthrough | tuned_family | `fair_leg_passthrough/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| fair_leg_passthrough | voltage_family | `fair_leg_passthrough/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2235 (-0.0000) | -0.2596 (+0.0001) | -0.4312 (+0.0012) | -0.1554 (-0.0003) |
| fair_leg_passthrough | voltage_off_matched | `fair_leg_passthrough/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2235 (+0.0000) | -0.2595 (+0.0001) | -0.4323 (+0.0001) | -0.1542 (+0.0009) |
| stress_export_default | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_export_default | fixed_base | `stress_export_default/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | -0.0691 (-0.2926) | -0.2311 (+0.0286) | -0.3003 (+0.1321) | -0.0939 (+0.0612) |
| stress_export_default | tuned_base | `stress_export_default/tuned_base` | 65.235 | 8.230 | 47.613 | 0.0065 | 300.222 | -0.0288 (-0.2523) | -0.2349 (+0.0248) | -0.3239 (+0.1084) | -0.1044 (+0.0507) |
| stress_export_default | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_export_default | tuned_family | `stress_export_default/tuned_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0135 (-0.2369) | -0.2181 (+0.0416) | -0.3273 (+0.1050) | -0.1154 (+0.0397) |
| stress_export_default | voltage_family | `stress_export_default/voltage_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0135 (-0.2369) | -0.2181 (+0.0416) | -0.3273 (+0.1050) | -0.1154 (+0.0397) |
| stress_export_default | voltage_off_matched | `stress_export_default/voltage_off_matched` | 60.829 | 8.028 | 13.695 | 0.0056 | 303.118 | -0.0181 (-0.2415) | -0.2209 (+0.0388) | -0.3280 (+0.1043) | -0.1133 (+0.0418) |
| stress_export_tuned | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_export_tuned | fixed_base | `stress_export_tuned/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | -0.0841 (-0.3076) | -0.2484 (+0.0113) | -0.2958 (+0.1365) | -0.0863 (+0.0689) |
| stress_export_tuned | tuned_base | `stress_export_tuned/tuned_base` | 60.770 | 8.229 | 43.657 | 0.0240 | 289.413 | 0.0620 (-0.1615) | -0.2458 (+0.0139) | -0.3487 (+0.0837) | -0.1159 (+0.0392) |
| stress_export_tuned | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_export_tuned | tuned_family | `stress_export_tuned/tuned_family` | 57.118 | 8.024 | 12.415 | 0.0204 | 293.958 | 0.1042 (-0.1193) | -0.2451 (+0.0146) | -0.3693 (+0.0630) | -0.1283 (+0.0268) |
| stress_export_tuned | voltage_family | `stress_export_tuned/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.0630 (-0.1605) | -0.2395 (+0.0202) | -0.3635 (+0.0689) | -0.1272 (+0.0279) |
| stress_export_tuned | voltage_off_matched | `stress_export_tuned/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.0199 (-0.2035) | -0.2414 (+0.0183) | -0.3518 (+0.0805) | -0.1137 (+0.0414) |
| stress_export_biting | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_export_biting | fixed_base | `stress_export_biting/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | -1.1493 (-1.3727) | -0.0694 (+0.1903) | 0.1559 (+0.5882) | 0.0917 (+0.2468) |
| stress_export_biting | tuned_base | `stress_export_biting/tuned_base` | 21.189 | 8.431 | 7.471 | 0.3845 | 68.781 | 0.2170 (-0.0065) | -0.0825 (+0.1772) | -0.1851 (+0.2472) | -0.0534 (+0.1017) |
| stress_export_biting | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_export_biting | tuned_family | `stress_export_biting/tuned_family` | 20.642 | 7.970 | 5.960 | 0.3876 | 69.630 | 0.2195 (-0.0040) | -0.0826 (+0.1770) | -0.1858 (+0.2465) | -0.0551 (+0.1000) |
| stress_export_biting | voltage_family | `stress_export_biting/voltage_family` | 37.301 | 8.153 | 6.850 | 0.1923 | 188.682 | -0.3454 (-0.5689) | 0.0104 (+0.2701) | -0.0732 (+0.3591) | -0.0278 (+0.1273) |
| stress_export_biting | voltage_off_matched | `stress_export_biting/voltage_off_matched` | 37.878 | 8.028 | 7.138 | 0.1868 | 191.961 | -0.3749 (-0.5984) | 0.0076 (+0.2673) | -0.0674 (+0.3649) | -0.0212 (+0.1339) |
| stress_bidirectional | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_bidirectional | fixed_base | `stress_bidirectional/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | -0.0677 (-0.2912) | -0.2288 (+0.0309) | -0.3028 (+0.1295) | -0.0934 (+0.0617) |
| stress_bidirectional | tuned_base | `stress_bidirectional/tuned_base` | 65.235 | 8.230 | 47.613 | 0.0065 | 300.222 | -0.0273 (-0.2507) | -0.2322 (+0.0275) | -0.3271 (+0.1052) | -0.1037 (+0.0514) |
| stress_bidirectional | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_bidirectional | tuned_family | `stress_bidirectional/tuned_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0119 (-0.2354) | -0.2156 (+0.0441) | -0.3301 (+0.1022) | -0.1149 (+0.0402) |
| stress_bidirectional | voltage_family | `stress_bidirectional/voltage_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0119 (-0.2354) | -0.2156 (+0.0441) | -0.3301 (+0.1022) | -0.1149 (+0.0402) |
| stress_bidirectional | voltage_off_matched | `stress_bidirectional/voltage_off_matched` | 60.829 | 8.028 | 13.695 | 0.0056 | 303.118 | -0.0164 (-0.2398) | -0.2179 (+0.0418) | -0.3313 (+0.1010) | -0.1127 (+0.0424) |
| stress_tenant_exempt | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_tenant_exempt | fixed_base | `stress_tenant_exempt/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | -0.0717 (-0.2951) | -0.2273 (+0.0324) | -0.3014 (+0.1310) | -0.0928 (+0.0623) |
| stress_tenant_exempt | tuned_base | `stress_tenant_exempt/tuned_base` | 65.235 | 8.230 | 47.613 | 0.0065 | 300.222 | -0.0321 (-0.2555) | -0.2304 (+0.0293) | -0.3253 (+0.1071) | -0.1030 (+0.0521) |
| stress_tenant_exempt | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_tenant_exempt | tuned_family | `stress_tenant_exempt/tuned_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0160 (-0.2395) | -0.2141 (+0.0456) | -0.3286 (+0.1038) | -0.1143 (+0.0409) |
| stress_tenant_exempt | voltage_family | `stress_tenant_exempt/voltage_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0160 (-0.2395) | -0.2141 (+0.0456) | -0.3286 (+0.1038) | -0.1143 (+0.0409) |
| stress_tenant_exempt | voltage_off_matched | `stress_tenant_exempt/voltage_off_matched` | 60.829 | 8.028 | 13.695 | 0.0056 | 303.118 | -0.0213 (-0.2447) | -0.2161 (+0.0436) | -0.3295 (+0.1028) | -0.1119 (+0.0432) |
| stress_smoothed | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_smoothed | fixed_base | `stress_smoothed/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.0065 (-0.2170) | -0.2484 (+0.0113) | -0.3339 (+0.0984) | -0.1046 (+0.0505) |
| stress_smoothed | tuned_base | `stress_smoothed/tuned_base` | 65.235 | 8.230 | 47.613 | 0.0065 | 300.222 | 0.0627 (-0.1608) | -0.2349 (+0.0248) | -0.3571 (+0.0752) | -0.1263 (+0.0288) |
| stress_smoothed | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_smoothed | tuned_family | `stress_smoothed/tuned_family` | 60.829 | 8.028 | 13.695 | 0.0056 | 303.118 | 0.0607 (-0.1628) | -0.2293 (+0.0304) | -0.3569 (+0.0755) | -0.1302 (+0.0249) |
| stress_smoothed | voltage_family | `stress_smoothed/voltage_family` | 61.114 | 7.911 | 13.126 | 0.0065 | 302.708 | 0.0616 (-0.1618) | -0.2323 (+0.0274) | -0.3565 (+0.0758) | -0.1297 (+0.0254) |
| stress_smoothed | voltage_off_matched | `stress_smoothed/voltage_off_matched` | 60.829 | 8.028 | 13.695 | 0.0056 | 303.118 | 0.0607 (-0.1628) | -0.2293 (+0.0304) | -0.3569 (+0.0755) | -0.1302 (+0.0249) |
| tou_midday_export | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| tou_midday_export | fixed_base | `tou_midday_export/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.0920 (-0.1314) | -0.2443 (+0.0154) | -0.3677 (+0.0646) | -0.1240 (+0.0311) |
| tou_midday_export | tuned_base | `tou_midday_export/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1089 (-0.1146) | -0.2294 (+0.0303) | -0.3843 (+0.0481) | -0.1353 (+0.0198) |
| tou_midday_export | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| tou_midday_export | tuned_family | `tou_midday_export/tuned_family` | 61.091 | 8.244 | 11.729 | 0.0042 | 304.056 | 0.1127 (-0.1108) | -0.2258 (+0.0338) | -0.3840 (+0.0483) | -0.1392 (+0.0159) |
| tou_midday_export | voltage_family | `tou_midday_export/voltage_family` | 59.978 | 8.073 | 12.313 | 0.0049 | 303.809 | 0.1118 (-0.1117) | -0.2265 (+0.0331) | -0.3828 (+0.0495) | -0.1391 (+0.0160) |
| tou_midday_export | voltage_off_matched | `tou_midday_export/voltage_off_matched` | 61.091 | 8.244 | 11.729 | 0.0042 | 304.056 | 0.1127 (-0.1108) | -0.2258 (+0.0338) | -0.3840 (+0.0483) | -0.1392 (+0.0159) |
| tou_evening_import | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| tou_evening_import | fixed_base | `tou_evening_import/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2202 (-0.0033) | -0.2541 (+0.0056) | -0.4220 (+0.0104) | -0.1479 (+0.0073) |
| tou_evening_import | tuned_base | `tou_evening_import/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2194 (-0.0041) | -0.2524 (+0.0073) | -0.4318 (+0.0006) | -0.1524 (+0.0027) |
| tou_evening_import | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| tou_evening_import | tuned_family | `tou_evening_import/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2206 (-0.0029) | -0.2509 (+0.0088) | -0.4338 (-0.0015) | -0.1547 (+0.0004) |
| tou_evening_import | voltage_family | `tou_evening_import/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2210 (-0.0025) | -0.2506 (+0.0091) | -0.4326 (-0.0003) | -0.1552 (-0.0001) |
| tou_evening_import | voltage_off_matched | `tou_evening_import/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2204 (-0.0031) | -0.2509 (+0.0088) | -0.4338 (-0.0014) | -0.1537 (+0.0015) |
| kva_coincident_charge | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| kva_coincident_charge | fixed_base | `kva_coincident_charge/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1143 (-0.1091) | -0.2596 (+0.0001) | -0.3769 (+0.0554) | -0.1256 (+0.0295) |
| kva_coincident_charge | tuned_base | `kva_coincident_charge/tuned_base` | 65.235 | 8.230 | 47.613 | 0.0065 | 300.222 | 0.1512 (-0.0723) | -0.2593 (+0.0004) | -0.3981 (+0.0342) | -0.1369 (+0.0182) |
| kva_coincident_charge | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| kva_coincident_charge | tuned_family | `kva_coincident_charge/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1753 (-0.0481) | -0.2543 (+0.0054) | -0.4108 (+0.0215) | -0.1469 (+0.0083) |
| kva_coincident_charge | voltage_family | `kva_coincident_charge/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1753 (-0.0481) | -0.2543 (+0.0054) | -0.4108 (+0.0215) | -0.1469 (+0.0083) |
| kva_coincident_charge | voltage_off_matched | `kva_coincident_charge/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.1582 (-0.0653) | -0.2548 (+0.0049) | -0.4065 (+0.0258) | -0.1410 (+0.0141) |
| kva_charge_class_neutral | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| kva_charge_class_neutral | fixed_base | `kva_charge_class_neutral/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.3004 (-0.0407) | -0.4178 (+0.0146) | -0.1422 (+0.0129) |
| kva_charge_class_neutral | tuned_base | `kva_charge_class_neutral/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2228 (-0.0006) | -0.3008 (-0.0411) | -0.4301 (+0.0023) | -0.1451 (+0.0100) |
| kva_charge_class_neutral | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| kva_charge_class_neutral | tuned_family | `kva_charge_class_neutral/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2905 (-0.0308) | -0.4317 (+0.0007) | -0.1492 (+0.0059) |
| kva_charge_class_neutral | voltage_family | `kva_charge_class_neutral/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2235 (-0.0000) | -0.2723 (-0.0127) | -0.4289 (+0.0035) | -0.1542 (+0.0009) |
| kva_charge_class_neutral | voltage_off_matched | `kva_charge_class_neutral/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2235 (+0.0000) | -0.2792 (-0.0195) | -0.4309 (+0.0014) | -0.1509 (+0.0042) |
| kva_peak_ratchet | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| kva_peak_ratchet | fixed_base | `kva_peak_ratchet/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2038 (-0.0197) | -0.2616 (-0.0019) | -0.4135 (+0.0188) | -0.1439 (+0.0112) |
| kva_peak_ratchet | tuned_base | `kva_peak_ratchet/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2019 (-0.0216) | -0.2603 (-0.0006) | -0.4228 (+0.0095) | -0.1482 (+0.0069) |
| kva_peak_ratchet | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| kva_peak_ratchet | tuned_family | `kva_peak_ratchet/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2114 (-0.0121) | -0.2587 (+0.0010) | -0.4266 (+0.0058) | -0.1529 (+0.0022) |
| kva_peak_ratchet | voltage_family | `kva_peak_ratchet/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2114 (-0.0121) | -0.2587 (+0.0010) | -0.4266 (+0.0058) | -0.1529 (+0.0022) |
| kva_peak_ratchet | voltage_off_matched | `kva_peak_ratchet/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2074 (-0.0161) | -0.2590 (+0.0007) | -0.4263 (+0.0060) | -0.1506 (+0.0045) |
| kw_peak_ratchet | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| kw_peak_ratchet | fixed_base | `kw_peak_ratchet/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2053 (-0.0182) | -0.2611 (-0.0014) | -0.4140 (+0.0184) | -0.1445 (+0.0106) |
| kw_peak_ratchet | tuned_base | `kw_peak_ratchet/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2034 (-0.0201) | -0.2600 (-0.0003) | -0.4234 (+0.0089) | -0.1486 (+0.0065) |
| kw_peak_ratchet | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| kw_peak_ratchet | tuned_family | `kw_peak_ratchet/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2110 (-0.0125) | -0.2584 (+0.0013) | -0.4266 (+0.0058) | -0.1527 (+0.0024) |
| kw_peak_ratchet | voltage_family | `kw_peak_ratchet/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2110 (-0.0125) | -0.2584 (+0.0013) | -0.4266 (+0.0058) | -0.1527 (+0.0024) |
| kw_peak_ratchet | voltage_off_matched | `kw_peak_ratchet/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2079 (-0.0156) | -0.2588 (+0.0009) | -0.4266 (+0.0058) | -0.1507 (+0.0044) |
| loss_share_linear | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| loss_share_linear | fixed_base | `loss_share_linear/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1904 (-0.0331) | -0.2596 (+0.0001) | -0.4074 (+0.0249) | -0.1419 (+0.0132) |
| loss_share_linear | tuned_base | `loss_share_linear/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1898 (-0.0336) | -0.2571 (+0.0026) | -0.4180 (+0.0143) | -0.1463 (+0.0089) |
| loss_share_linear | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| loss_share_linear | tuned_family | `loss_share_linear/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1948 (-0.0287) | -0.2544 (+0.0053) | -0.4198 (+0.0125) | -0.1503 (+0.0049) |
| loss_share_linear | voltage_family | `loss_share_linear/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1948 (-0.0287) | -0.2544 (+0.0053) | -0.4198 (+0.0125) | -0.1503 (+0.0049) |
| loss_share_linear | voltage_off_matched | `loss_share_linear/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.1930 (-0.0305) | -0.2548 (+0.0049) | -0.4207 (+0.0117) | -0.1482 (+0.0069) |
| loss_share_quadratic | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| loss_share_quadratic | fixed_base | `loss_share_quadratic/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1865 (-0.0370) | -0.2681 (-0.0084) | -0.4074 (+0.0249) | -0.1383 (+0.0168) |
| loss_share_quadratic | tuned_base | `loss_share_quadratic/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1858 (-0.0377) | -0.2643 (-0.0046) | -0.4178 (+0.0145) | -0.1430 (+0.0121) |
| loss_share_quadratic | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| loss_share_quadratic | tuned_family | `loss_share_quadratic/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1909 (-0.0326) | -0.2596 (+0.0001) | -0.4182 (+0.0141) | -0.1483 (+0.0068) |
| loss_share_quadratic | voltage_family | `loss_share_quadratic/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1909 (-0.0326) | -0.2596 (+0.0001) | -0.4182 (+0.0141) | -0.1483 (+0.0068) |
| loss_share_quadratic | voltage_off_matched | `loss_share_quadratic/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.1889 (-0.0346) | -0.2602 (-0.0005) | -0.4192 (+0.0132) | -0.1461 (+0.0090) |
| voltage_level_exposure | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| voltage_level_exposure | fixed_base | `voltage_level_exposure/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.0984 (-0.1251) | -0.3336 (-0.0739) | -0.4067 (+0.0257) | -0.0844 (+0.0707) |
| voltage_level_exposure | tuned_base | `voltage_level_exposure/tuned_base` | 65.235 | 8.230 | 47.613 | 0.0065 | 300.222 | 0.1155 (-0.1080) | -0.3225 (-0.0628) | -0.4171 (+0.0152) | -0.0955 (+0.0596) |
| voltage_level_exposure | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| voltage_level_exposure | tuned_family | `voltage_level_exposure/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1282 (-0.0952) | -0.3130 (-0.0533) | -0.4204 (+0.0119) | -0.1069 (+0.0482) |
| voltage_level_exposure | voltage_family | `voltage_level_exposure/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1282 (-0.0952) | -0.3130 (-0.0533) | -0.4204 (+0.0119) | -0.1069 (+0.0482) |
| voltage_level_exposure | voltage_off_matched | `voltage_level_exposure/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.1194 (-0.1041) | -0.3189 (-0.0592) | -0.4231 (+0.0092) | -0.0995 (+0.0557) |
| voltage_sensitivity | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| voltage_sensitivity | fixed_base | `voltage_sensitivity/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1390 (-0.0845) | -0.2475 (+0.0122) | -0.4054 (+0.0270) | -0.1218 (+0.0333) |
| voltage_sensitivity | tuned_base | `voltage_sensitivity/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1465 (-0.0770) | -0.2705 (-0.0108) | -0.4349 (-0.0026) | -0.1130 (+0.0421) |
| voltage_sensitivity | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| voltage_sensitivity | tuned_family | `voltage_sensitivity/tuned_family` | 64.196 | 7.763 | 14.817 | 0.0131 | 298.712 | 0.1710 (-0.0525) | -0.2293 (+0.0304) | -0.4283 (+0.0040) | -0.1317 (+0.0234) |
| voltage_sensitivity | voltage_family | `voltage_sensitivity/voltage_family` | 59.559 | 8.310 | 11.744 | 0.0055 | 303.454 | 0.1649 (-0.0586) | -0.2602 (-0.0005) | -0.4388 (-0.0064) | -0.1219 (+0.0332) |
| voltage_sensitivity | voltage_off_matched | `voltage_sensitivity/voltage_off_matched` | 62.296 | 8.471 | 9.329 | 0.0036 | 304.415 | 0.1525 (-0.0710) | -0.2532 (+0.0065) | -0.4359 (-0.0036) | -0.1206 (+0.0345) |
| staggered_class_tou | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| staggered_class_tou | fixed_base | `staggered_class_tou/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1322 (-0.0913) | -0.2474 (+0.0122) | -0.3845 (+0.0479) | -0.1316 (+0.0235) |
| staggered_class_tou | tuned_base | `staggered_class_tou/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1506 (-0.0729) | -0.2313 (+0.0284) | -0.3980 (+0.0344) | -0.1458 (+0.0093) |
| staggered_class_tou | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| staggered_class_tou | tuned_family | `staggered_class_tou/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.1506 (-0.0729) | -0.2307 (+0.0290) | -0.4000 (+0.0324) | -0.1473 (+0.0078) |
| staggered_class_tou | voltage_family | `staggered_class_tou/voltage_family` | 59.978 | 8.073 | 12.313 | 0.0049 | 303.809 | 0.1497 (-0.0738) | -0.2317 (+0.0280) | -0.3976 (+0.0347) | -0.1464 (+0.0087) |
| staggered_class_tou | voltage_off_matched | `staggered_class_tou/voltage_off_matched` | 61.091 | 8.244 | 11.729 | 0.0042 | 304.056 | 0.1518 (-0.0717) | -0.2302 (+0.0295) | -0.3991 (+0.0332) | -0.1471 (+0.0080) |
| own_peak_ratchet | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| own_peak_ratchet | fixed_base | `own_peak_ratchet/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2064 (-0.0170) | -0.2651 (-0.0054) | -0.4147 (+0.0177) | -0.1437 (+0.0114) |
| own_peak_ratchet | tuned_base | `own_peak_ratchet/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2061 (-0.0174) | -0.2631 (-0.0034) | -0.4244 (+0.0079) | -0.1486 (+0.0065) |
| own_peak_ratchet | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| own_peak_ratchet | tuned_family | `own_peak_ratchet/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2072 (-0.0163) | -0.2616 (-0.0019) | -0.4265 (+0.0058) | -0.1507 (+0.0044) |
| own_peak_ratchet | voltage_family | `own_peak_ratchet/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2082 (-0.0152) | -0.2608 (-0.0011) | -0.4253 (+0.0070) | -0.1517 (+0.0034) |
| own_peak_ratchet | voltage_off_matched | `own_peak_ratchet/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2080 (-0.0155) | -0.2609 (-0.0012) | -0.4266 (+0.0057) | -0.1502 (+0.0049) |
| own_deviation_charge | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| own_deviation_charge | fixed_base | `own_deviation_charge/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1440 (-0.0795) | -0.2435 (+0.0162) | -0.3909 (+0.0415) | -0.1340 (+0.0212) |
| own_deviation_charge | tuned_base | `own_deviation_charge/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1395 (-0.0840) | -0.2446 (+0.0151) | -0.4030 (+0.0293) | -0.1348 (+0.0203) |
| own_deviation_charge | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| own_deviation_charge | tuned_family | `own_deviation_charge/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1496 (-0.0738) | -0.2363 (+0.0234) | -0.4039 (+0.0284) | -0.1429 (+0.0122) |
| own_deviation_charge | voltage_family | `own_deviation_charge/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1496 (-0.0738) | -0.2363 (+0.0234) | -0.4039 (+0.0284) | -0.1429 (+0.0122) |
| own_deviation_charge | voltage_off_matched | `own_deviation_charge/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.1480 (-0.0754) | -0.2374 (+0.0222) | -0.4061 (+0.0262) | -0.1400 (+0.0151) |
| reverse_flow_charge | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| reverse_flow_charge | fixed_base | `reverse_flow_charge/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1161 (-0.1073) | -0.2361 (+0.0236) | -0.3751 (+0.0572) | -0.1323 (+0.0228) |
| reverse_flow_charge | tuned_base | `reverse_flow_charge/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1109 (-0.1126) | -0.2357 (+0.0240) | -0.3849 (+0.0475) | -0.1346 (+0.0206) |
| reverse_flow_charge | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| reverse_flow_charge | tuned_family | `reverse_flow_charge/tuned_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1129 (-0.1106) | -0.2335 (+0.0261) | -0.3857 (+0.0466) | -0.1376 (+0.0175) |
| reverse_flow_charge | voltage_family | `reverse_flow_charge/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1129 (-0.1106) | -0.2335 (+0.0261) | -0.3857 (+0.0466) | -0.1376 (+0.0175) |
| reverse_flow_charge | voltage_off_matched | `reverse_flow_charge/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.1105 (-0.1130) | -0.2353 (+0.0244) | -0.3870 (+0.0454) | -0.1348 (+0.0203) |
| two_part_volumetric | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| two_part_volumetric | fixed_base | `two_part_volumetric/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.1786 (-0.0449) | -0.2459 (+0.0138) | -0.4044 (+0.0279) | -0.1411 (+0.0140) |
| two_part_volumetric | tuned_base | `two_part_volumetric/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.1752 (-0.0483) | -0.2462 (+0.0135) | -0.4143 (+0.0181) | -0.1440 (+0.0111) |
| two_part_volumetric | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| two_part_volumetric | tuned_family | `two_part_volumetric/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.1767 (-0.0468) | -0.2444 (+0.0153) | -0.4163 (+0.0160) | -0.1465 (+0.0086) |
| two_part_volumetric | voltage_family | `two_part_volumetric/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.1778 (-0.0456) | -0.2435 (+0.0162) | -0.4152 (+0.0171) | -0.1475 (+0.0076) |
| two_part_volumetric | voltage_off_matched | `two_part_volumetric/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.1758 (-0.0477) | -0.2450 (+0.0147) | -0.4165 (+0.0158) | -0.1448 (+0.0103) |
| two_part_fixed | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| two_part_fixed | fixed_base | `two_part_fixed/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2677 (+0.0442) | -0.2797 (-0.0200) | -0.4368 (-0.0045) | -0.1554 (-0.0003) |
| two_part_fixed | tuned_base | `two_part_fixed/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2705 (+0.0470) | -0.2753 (-0.0156) | -0.4465 (-0.0141) | -0.1623 (-0.0072) |
| two_part_fixed | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| two_part_fixed | tuned_family | `two_part_fixed/tuned_family` | 63.937 | 10.128 | 6.194 | 0.0029 | 304.153 | 0.2722 (+0.0487) | -0.2733 (-0.0136) | -0.4482 (-0.0159) | -0.1638 (-0.0087) |
| two_part_fixed | voltage_family | `two_part_fixed/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2691 (+0.0456) | -0.2758 (-0.0161) | -0.4471 (-0.0148) | -0.1634 (-0.0083) |
| two_part_fixed | voltage_off_matched | `two_part_fixed/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2712 (+0.0477) | -0.2741 (-0.0144) | -0.4480 (-0.0156) | -0.1636 (-0.0085) |
| tenant_floor | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| tenant_floor | fixed_base | `tenant_floor/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2020 (-0.0215) | -0.2549 (+0.0048) | -0.4127 (+0.0197) | -0.1450 (+0.0101) |
| tenant_floor | tuned_base | `tenant_floor/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2017 (-0.0218) | -0.2528 (+0.0068) | -0.4224 (+0.0099) | -0.1499 (+0.0052) |
| tenant_floor | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| tenant_floor | tuned_family | `tenant_floor/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2023 (-0.0212) | -0.2518 (+0.0079) | -0.4244 (+0.0079) | -0.1518 (+0.0033) |
| tenant_floor | voltage_family | `tenant_floor/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2023 (-0.0212) | -0.2517 (+0.0080) | -0.4232 (+0.0091) | -0.1522 (+0.0030) |
| tenant_floor | voltage_off_matched | `tenant_floor/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2023 (-0.0212) | -0.2516 (+0.0081) | -0.4243 (+0.0080) | -0.1509 (+0.0042) |
| stress_funds_tenant_floor | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_funds_tenant_floor | fixed_base | `stress_funds_tenant_floor/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | -0.6536 (-0.8771) | -0.0127 (+0.2470) | -0.0809 (+0.3515) | -0.0040 (+0.1511) |
| stress_funds_tenant_floor | tuned_base | `stress_funds_tenant_floor/tuned_base` | 39.040 | 8.229 | 24.613 | 0.1911 | 187.038 | 0.1051 (-0.1184) | -0.1588 (+0.1008) | -0.2645 (+0.1678) | -0.0813 (+0.0738) |
| stress_funds_tenant_floor | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_funds_tenant_floor | tuned_family | `stress_funds_tenant_floor/tuned_family` | 37.878 | 8.028 | 7.138 | 0.1868 | 191.961 | 0.1159 (-0.1076) | -0.1607 (+0.0990) | -0.2718 (+0.1606) | -0.0874 (+0.0677) |
| stress_funds_tenant_floor | voltage_family | `stress_funds_tenant_floor/voltage_family` | 37.301 | 8.153 | 6.850 | 0.1923 | 188.682 | 0.1269 (-0.0966) | -0.1640 (+0.0957) | -0.2718 (+0.1605) | -0.0877 (+0.0674) |
| stress_funds_tenant_floor | voltage_off_matched | `stress_funds_tenant_floor/voltage_off_matched` | 37.878 | 8.028 | 7.138 | 0.1868 | 191.961 | 0.1159 (-0.1076) | -0.1607 (+0.0990) | -0.2718 (+0.1606) | -0.0874 (+0.0677) |
| stress_class_neutral | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| stress_class_neutral | fixed_base | `stress_class_neutral/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.3403 (-0.0806) | -0.4100 (+0.0224) | -0.1388 (+0.0163) |
| stress_class_neutral | tuned_base | `stress_class_neutral/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 303.018 | 0.2228 (-0.0006) | -0.3422 (-0.0825) | -0.4293 (+0.0031) | -0.1370 (+0.0181) |
| stress_class_neutral | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| stress_class_neutral | tuned_family | `stress_class_neutral/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.3279 (-0.0682) | -0.4296 (+0.0027) | -0.1427 (+0.0124) |
| stress_class_neutral | voltage_family | `stress_class_neutral/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 304.859 | 0.2235 (-0.0000) | -0.3053 (-0.0456) | -0.4207 (+0.0116) | -0.1524 (+0.0027) |
| stress_class_neutral | voltage_off_matched | `stress_class_neutral/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 304.390 | 0.2235 (+0.0000) | -0.3152 (-0.0555) | -0.4261 (+0.0062) | -0.1465 (+0.0086) |
| joint_system_cost | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| joint_system_cost | fixed_base | `joint_system_cost/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | -0.1437 (-0.3671) | -0.2308 (+0.0289) | -0.2731 (+0.1592) | -0.0761 (+0.0790) |
| joint_system_cost | tuned_base | `joint_system_cost/tuned_base` | 65.235 | 8.230 | 47.613 | 0.0065 | 300.222 | -0.1002 (-0.3237) | -0.2313 (+0.0284) | -0.2981 (+0.1342) | -0.0880 (+0.0671) |
| joint_system_cost | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| joint_system_cost | tuned_family | `joint_system_cost/tuned_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0824 (-0.3059) | -0.2129 (+0.0468) | -0.3016 (+0.1308) | -0.1004 (+0.0547) |
| joint_system_cost | voltage_family | `joint_system_cost/voltage_family` | 61.847 | 7.814 | 13.660 | 0.0072 | 302.234 | -0.0824 (-0.3059) | -0.2129 (+0.0468) | -0.3016 (+0.1308) | -0.1004 (+0.0547) |
| joint_system_cost | voltage_off_matched | `joint_system_cost/voltage_off_matched` | 60.829 | 8.028 | 13.695 | 0.0056 | 303.118 | -0.0872 (-0.3107) | -0.2158 (+0.0439) | -0.3023 (+0.1300) | -0.0982 (+0.0569) |
| subsidy_ablation | reference_default_base | `fair_leg/default_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 294.280 | 0.2231 (-0.0003) | -0.2628 (-0.0031) | -0.4206 (+0.0117) | -0.1483 (+0.0069) |
| subsidy_ablation | fixed_base | `subsidy_ablation/fixed_base` | 65.305 | 7.895 | 15.280 | 0.0150 | 536.200 | 0.0114 (-0.2121) | -0.4210 (-0.1613) | -0.5796 (-0.1473) | -0.2138 (-0.0587) |
| subsidy_ablation | tuned_base | `subsidy_ablation/tuned_base` | 67.194 | 8.889 | 48.412 | 0.0031 | 544.938 | 0.0111 (-0.2124) | -0.4190 (-0.1593) | -0.5894 (-0.1570) | -0.2187 (-0.0636) |
| subsidy_ablation | reference_tuned_family | `fair_leg/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 305.195 | 0.2235 (+0.0000) | -0.2597 (+0.0000) | -0.4323 (+0.0000) | -0.1551 (+0.0000) |
| subsidy_ablation | tuned_family | `subsidy_ablation/tuned_family` | 65.758 | 8.841 | 9.807 | 0.0029 | 547.115 | 0.0117 (-0.2117) | -0.4179 (-0.1582) | -0.5913 (-0.1590) | -0.2207 (-0.0656) |
| subsidy_ablation | voltage_family | `subsidy_ablation/voltage_family` | 59.199 | 8.257 | 6.855 | 0.0038 | 546.779 | 0.0117 (-0.2117) | -0.4179 (-0.1582) | -0.5901 (-0.1578) | -0.2210 (-0.0659) |
| subsidy_ablation | voltage_off_matched | `subsidy_ablation/voltage_off_matched` | 62.743 | 9.463 | 6.146 | 0.0029 | 546.310 | 0.0117 (-0.2117) | -0.4178 (-0.1581) | -0.5912 (-0.1589) | -0.2198 (-0.0647) |

## Per-tariff validation Pareto sets

These are the runner's tariff-grouped engineering frontiers within each economically credible shared-policy response set.

| Tariff | Policy id | Controller | Training settlement gap CHF/agent |
|---|---:|---|---:|
| fair_leg | 4 | delay_13 | 0.000000 |
| fair_leg | 7 | slow_stagger | 0.036915 |
| fair_leg | 31 | instant_slow_stagger_1h | 0.038048 |
| fair_leg | 6 | stagger_11_2h | 0.060707 |
| fair_leg | 5 | stagger_11_1h | 0.085384 |
| fair_leg | 28 | instant_stagger_2h_10 | 0.094246 |
| fair_leg_passthrough | 4 | delay_13 | 0.000000 |
| fair_leg_passthrough | 7 | slow_stagger | 0.036915 |
| fair_leg_passthrough | 31 | instant_slow_stagger_1h | 0.038048 |
| fair_leg_passthrough | 6 | stagger_11_2h | 0.060707 |
| fair_leg_passthrough | 5 | stagger_11_1h | 0.085384 |
| fair_leg_passthrough | 28 | instant_stagger_2h_10 | 0.094246 |
| joint_system_cost | 23 | instant_10 | 0.000000 |
| joint_system_cost | 29 | instant_stagger_1h_10_ref101 | 0.032969 |
| joint_system_cost | 22 | instant_5 | 0.034262 |
| joint_system_cost | 26 | instant_stagger_1h_10 | 0.066732 |
| joint_system_cost | 3 | delay_11 | 0.073454 |
| joint_system_cost | 25 | instant_stagger_1h_5 | 0.073469 |
| joint_system_cost | 24 | instant_25 | 0.075747 |
| joint_system_cost | 5 | stagger_11_1h | 0.087013 |
| joint_system_cost | 28 | instant_stagger_2h_10 | 0.093748 |
| kva_charge_class_neutral | 4 | delay_13 | 0.000000 |
| kva_charge_class_neutral | 7 | slow_stagger | 0.036919 |
| kva_charge_class_neutral | 31 | instant_slow_stagger_1h | 0.038048 |
| kva_charge_class_neutral | 6 | stagger_11_2h | 0.060707 |
| kva_charge_class_neutral | 5 | stagger_11_1h | 0.085384 |
| kva_charge_class_neutral | 28 | instant_stagger_2h_10 | 0.094246 |
| kva_coincident_charge | 31 | instant_slow_stagger_1h | 0.000000 |
| kva_peak_ratchet | 31 | instant_slow_stagger_1h | 0.000000 |
| kva_peak_ratchet | 28 | instant_stagger_2h_10 | 0.086784 |
| kw_peak_ratchet | 31 | instant_slow_stagger_1h | 0.000000 |
| kw_peak_ratchet | 28 | instant_stagger_2h_10 | 0.068455 |
| loss_share_linear | 31 | instant_slow_stagger_1h | 0.000000 |
| loss_share_linear | 6 | stagger_11_2h | 0.036247 |
| loss_share_linear | 28 | instant_stagger_2h_10 | 0.043194 |
| loss_share_linear | 5 | stagger_11_1h | 0.046539 |
| loss_share_linear | 4 | delay_13 | 0.057339 |
| loss_share_linear | 25 | instant_stagger_1h_5 | 0.058060 |
| loss_share_linear | 29 | instant_stagger_1h_10_ref101 | 0.064907 |
| loss_share_linear | 26 | instant_stagger_1h_10 | 0.085743 |
| loss_share_linear | 3 | delay_11 | 0.086296 |
| loss_share_linear | 7 | slow_stagger | 0.093193 |
| loss_share_quadratic | 31 | instant_slow_stagger_1h | 0.000000 |
| loss_share_quadratic | 6 | stagger_11_2h | 0.035847 |
| loss_share_quadratic | 28 | instant_stagger_2h_10 | 0.042080 |
| loss_share_quadratic | 5 | stagger_11_1h | 0.045258 |
| loss_share_quadratic | 25 | instant_stagger_1h_5 | 0.056568 |
| loss_share_quadratic | 4 | delay_13 | 0.058723 |
| loss_share_quadratic | 29 | instant_stagger_1h_10_ref101 | 0.063225 |
| loss_share_quadratic | 26 | instant_stagger_1h_10 | 0.084095 |
| loss_share_quadratic | 3 | delay_11 | 0.084774 |
| loss_share_quadratic | 7 | slow_stagger | 0.099197 |
| own_deviation_charge | 31 | instant_slow_stagger_1h | 0.000000 |
| own_deviation_charge | 7 | slow_stagger | 0.091314 |
| own_peak_ratchet | 4 | delay_13 | 0.000000 |
| own_peak_ratchet | 31 | instant_slow_stagger_1h | 0.005489 |
| own_peak_ratchet | 7 | slow_stagger | 0.017216 |
| own_peak_ratchet | 6 | stagger_11_2h | 0.056828 |
| own_peak_ratchet | 5 | stagger_11_1h | 0.078285 |
| own_peak_ratchet | 28 | instant_stagger_2h_10 | 0.085087 |
| reverse_flow_charge | 31 | instant_slow_stagger_1h | 0.000000 |
| reverse_flow_charge | 4 | delay_13 | 0.002174 |
| reverse_flow_charge | 6 | stagger_11_2h | 0.025347 |
| reverse_flow_charge | 28 | instant_stagger_2h_10 | 0.033401 |
| reverse_flow_charge | 5 | stagger_11_1h | 0.035482 |
| reverse_flow_charge | 29 | instant_stagger_1h_10_ref101 | 0.053928 |
| reverse_flow_charge | 26 | instant_stagger_1h_10 | 0.070816 |
| staggered_class_tou | 4 | delay_13 | 0.000000 |
| staggered_class_tou | 6 | stagger_11_2h | 0.022036 |
| staggered_class_tou | 5 | stagger_11_1h | 0.058884 |
| stress_bidirectional | 23 | instant_10 | 0.000000 |
| stress_bidirectional | 22 | instant_5 | 0.030348 |
| stress_bidirectional | 29 | instant_stagger_1h_10_ref101 | 0.042067 |
| stress_bidirectional | 3 | delay_11 | 0.067194 |
| stress_bidirectional | 24 | instant_25 | 0.067972 |
| stress_bidirectional | 26 | instant_stagger_1h_10 | 0.072676 |
| stress_bidirectional | 25 | instant_stagger_1h_5 | 0.073975 |
| stress_bidirectional | 5 | stagger_11_1h | 0.075699 |
| stress_class_neutral | 4 | delay_13 | 0.000000 |
| stress_class_neutral | 7 | slow_stagger | 0.036911 |
| stress_class_neutral | 31 | instant_slow_stagger_1h | 0.038048 |
| stress_class_neutral | 6 | stagger_11_2h | 0.060703 |
| stress_class_neutral | 5 | stagger_11_1h | 0.085381 |
| stress_class_neutral | 28 | instant_stagger_2h_10 | 0.094242 |
| stress_export_biting | 12 | cap_2 | 0.000000 |
| stress_export_default | 23 | instant_10 | 0.000000 |
| stress_export_default | 22 | instant_5 | 0.032106 |
| stress_export_default | 29 | instant_stagger_1h_10_ref101 | 0.043211 |
| stress_export_default | 24 | instant_25 | 0.066732 |
| stress_export_default | 3 | delay_11 | 0.072092 |
| stress_export_default | 26 | instant_stagger_1h_10 | 0.073166 |
| stress_export_default | 25 | instant_stagger_1h_5 | 0.075735 |
| stress_export_default | 5 | stagger_11_1h | 0.082262 |
| stress_export_tuned | 10 | cap_8 | 0.000000 |
| stress_funds_tenant_floor | 11 | cap_4 | 0.000000 |
| stress_funds_tenant_floor | 17 | voltage_cap | 0.036554 |
| stress_smoothed | 3 | delay_11 | 0.000000 |
| stress_smoothed | 22 | instant_5 | 0.035627 |
| stress_tenant_exempt | 23 | instant_10 | 0.000000 |
| stress_tenant_exempt | 22 | instant_5 | 0.034847 |
| stress_tenant_exempt | 29 | instant_stagger_1h_10_ref101 | 0.047890 |
| stress_tenant_exempt | 24 | instant_25 | 0.064301 |
| stress_tenant_exempt | 26 | instant_stagger_1h_10 | 0.075169 |
| stress_tenant_exempt | 3 | delay_11 | 0.081535 |
| stress_tenant_exempt | 25 | instant_stagger_1h_5 | 0.082296 |
| stress_tenant_exempt | 5 | stagger_11_1h | 0.096483 |
| subsidy_ablation | 4 | delay_13 | 0.000000 |
| subsidy_ablation | 7 | slow_stagger | 0.036911 |
| subsidy_ablation | 31 | instant_slow_stagger_1h | 0.038044 |
| subsidy_ablation | 6 | stagger_11_2h | 0.060703 |
| subsidy_ablation | 5 | stagger_11_1h | 0.085381 |
| subsidy_ablation | 28 | instant_stagger_2h_10 | 0.094242 |
| tenant_floor | 4 | delay_13 | 0.000000 |
| tenant_floor | 7 | slow_stagger | 0.036915 |
| tenant_floor | 31 | instant_slow_stagger_1h | 0.038048 |
| tenant_floor | 6 | stagger_11_2h | 0.060703 |
| tenant_floor | 5 | stagger_11_1h | 0.085384 |
| tenant_floor | 28 | instant_stagger_2h_10 | 0.094246 |
| tou_evening_import | 4 | delay_13 | 0.000000 |
| tou_evening_import | 31 | instant_slow_stagger_1h | 0.032692 |
| tou_evening_import | 7 | slow_stagger | 0.046837 |
| tou_evening_import | 6 | stagger_11_2h | 0.057507 |
| tou_evening_import | 5 | stagger_11_1h | 0.080585 |
| tou_evening_import | 28 | instant_stagger_2h_10 | 0.088768 |
| tou_midday_export | 5 | stagger_11_1h | 0.000000 |
| tou_midday_export | 6 | stagger_11_2h | 0.003706 |
| tou_midday_export | 4 | delay_13 | 0.028847 |
| tou_midday_export | 3 | delay_11 | 0.035534 |
| tou_midday_export | 25 | instant_stagger_1h_5 | 0.060963 |
| tou_midday_export | 29 | instant_stagger_1h_10_ref101 | 0.089876 |
| tou_midday_export | 31 | instant_slow_stagger_1h | 0.093874 |
| tou_midday_export | 28 | instant_stagger_2h_10 | 0.099728 |
| two_part_fixed | 7 | slow_stagger | 0.000000 |
| two_part_fixed | 4 | delay_13 | 0.017460 |
| two_part_fixed | 31 | instant_slow_stagger_1h | 0.088566 |
| two_part_volumetric | 4 | delay_13 | 0.000000 |
| two_part_volumetric | 31 | instant_slow_stagger_1h | 0.004990 |
| two_part_volumetric | 6 | stagger_11_2h | 0.030350 |
| two_part_volumetric | 28 | instant_stagger_2h_10 | 0.043095 |
| two_part_volumetric | 5 | stagger_11_1h | 0.043255 |
| two_part_volumetric | 25 | instant_stagger_1h_5 | 0.055805 |
| two_part_volumetric | 29 | instant_stagger_1h_10_ref101 | 0.065430 |
| two_part_volumetric | 26 | instant_stagger_1h_10 | 0.083324 |
| two_part_volumetric | 7 | slow_stagger | 0.091290 |
| voltage_level_exposure | 31 | instant_slow_stagger_1h | 0.000000 |
| voltage_level_exposure | 28 | instant_stagger_2h_10 | 0.069141 |
| voltage_level_exposure | 29 | instant_stagger_1h_10_ref101 | 0.073723 |
| voltage_level_exposure | 26 | instant_stagger_1h_10 | 0.099285 |
| voltage_sensitivity | 2 | delay_09 | 0.000000 |
| voltage_sensitivity | 15 | voltage_stagger | 0.099762 |

## Cross-tariff validation Pareto set

The mask was applied once across all validation rows in `credible_response_candidates.csv`, without grouping by tariff.

| Tariff | Policy id | Controller | Training settlement gap CHF/agent |
|---|---:|---|---:|
| joint_system_cost | 23 | instant_10 | 0.000000 |
| joint_system_cost | 29 | instant_stagger_1h_10_ref101 | 0.032969 |
| joint_system_cost | 22 | instant_5 | 0.034262 |
| joint_system_cost | 26 | instant_stagger_1h_10 | 0.066732 |
| joint_system_cost | 3 | delay_11 | 0.073454 |
| joint_system_cost | 25 | instant_stagger_1h_5 | 0.073469 |
| joint_system_cost | 24 | instant_25 | 0.075747 |
| joint_system_cost | 5 | stagger_11_1h | 0.087013 |
| joint_system_cost | 28 | instant_stagger_2h_10 | 0.093748 |
| loss_share_linear | 29 | instant_stagger_1h_10_ref101 | 0.064907 |
| loss_share_linear | 26 | instant_stagger_1h_10 | 0.085743 |
| loss_share_quadratic | 29 | instant_stagger_1h_10_ref101 | 0.063225 |
| loss_share_quadratic | 26 | instant_stagger_1h_10 | 0.084095 |
| reverse_flow_charge | 29 | instant_stagger_1h_10_ref101 | 0.053928 |
| reverse_flow_charge | 26 | instant_stagger_1h_10 | 0.070816 |
| stress_bidirectional | 23 | instant_10 | 0.000000 |
| stress_bidirectional | 22 | instant_5 | 0.030348 |
| stress_bidirectional | 29 | instant_stagger_1h_10_ref101 | 0.042067 |
| stress_bidirectional | 3 | delay_11 | 0.067194 |
| stress_bidirectional | 24 | instant_25 | 0.067972 |
| stress_bidirectional | 26 | instant_stagger_1h_10 | 0.072676 |
| stress_bidirectional | 25 | instant_stagger_1h_5 | 0.073975 |
| stress_bidirectional | 5 | stagger_11_1h | 0.075699 |
| stress_export_biting | 12 | cap_2 | 0.000000 |
| stress_export_default | 23 | instant_10 | 0.000000 |
| stress_export_default | 22 | instant_5 | 0.032106 |
| stress_export_default | 29 | instant_stagger_1h_10_ref101 | 0.043211 |
| stress_export_default | 24 | instant_25 | 0.066732 |
| stress_export_default | 3 | delay_11 | 0.072092 |
| stress_export_default | 26 | instant_stagger_1h_10 | 0.073166 |
| stress_export_default | 25 | instant_stagger_1h_5 | 0.075735 |
| stress_export_default | 5 | stagger_11_1h | 0.082262 |
| stress_export_tuned | 10 | cap_8 | 0.000000 |
| stress_funds_tenant_floor | 11 | cap_4 | 0.000000 |
| stress_funds_tenant_floor | 17 | voltage_cap | 0.036554 |
| stress_tenant_exempt | 23 | instant_10 | 0.000000 |
| stress_tenant_exempt | 22 | instant_5 | 0.034847 |
| stress_tenant_exempt | 29 | instant_stagger_1h_10_ref101 | 0.047890 |
| stress_tenant_exempt | 24 | instant_25 | 0.064301 |
| stress_tenant_exempt | 26 | instant_stagger_1h_10 | 0.075169 |
| stress_tenant_exempt | 3 | delay_11 | 0.081535 |
| stress_tenant_exempt | 25 | instant_stagger_1h_5 | 0.082296 |
| stress_tenant_exempt | 5 | stagger_11_1h | 0.096483 |
| subsidy_ablation | 4 | delay_13 | 0.000000 |
| subsidy_ablation | 7 | slow_stagger | 0.036911 |
| subsidy_ablation | 31 | instant_slow_stagger_1h | 0.038044 |
| subsidy_ablation | 6 | stagger_11_2h | 0.060703 |
| subsidy_ablation | 5 | stagger_11_1h | 0.085381 |
| subsidy_ablation | 28 | instant_stagger_2h_10 | 0.094242 |
| tou_midday_export | 29 | instant_stagger_1h_10_ref101 | 0.089876 |
| two_part_volumetric | 29 | instant_stagger_1h_10_ref101 | 0.065430 |
| two_part_volumetric | 26 | instant_stagger_1h_10 | 0.083324 |
| voltage_level_exposure | 29 | instant_stagger_1h_10_ref101 | 0.073723 |
| voltage_level_exposure | 26 | instant_stagger_1h_10 | 0.099285 |
| voltage_sensitivity | 2 | delay_09 | 0.000000 |
| voltage_sensitivity | 15 | voltage_stagger | 0.099762 |

## Limitations

- Controller tuning is a shared-policy approximation, not proof of an individual equilibrium.
- Controllers observe neither live prices nor neighbours; tariff effects enter through ex-ante policy tuning and settlement replay.
- Both Pareto sets use validation weather only. Test weather did not select controllers or construct or revise either frontier.
- The official `score()`/`evaluate()` submission path was not used as a per-tariff scorer. The runner retained the unchanged jury metrics while using its controller-family experiment pipeline.
- `accepted: false` identifies a diagnostic finalist, not a recommended tariff.
- Battery wear and terminal stored-energy value are absent from the household objective; revenue neutrality alone does not establish fairness.
