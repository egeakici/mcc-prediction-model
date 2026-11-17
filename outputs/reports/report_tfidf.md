# TF-IDF MCC

## Veri

| source_id   | raw_rows   | excluded_rows   | no_code_rows   | remaining_rows   | model_records   | unique_texts   | unique_key_loose   | unique_codes   |
|-------------|------------|-----------------|----------------|------------------|-----------------|----------------|--------------------|----------------|
| oklahoma    | 6883868    | 2013965         | 1830676        | 4869903          | 181810          | 167558         | 167164             | 254            |
| delaware    | 2251316    | 681862          | 669489         | 1569454          | 86203           | 82089          | 82036              | 206            |
| sunderland  | 255169     | 41767           | 41453          | 213402           | 3132            | 3034           | 2926               | 171            |

## Dışlamalar (sıralı)

| source_id   | reason              | n_rows   |
|-------------|---------------------|----------|
| oklahoma    | record_type         | 102659   |
| oklahoma    | merchant_masked     | 0        |
| oklahoma    | no_code             | 1830676  |
| oklahoma    | linked_needs_review | 0        |
| oklahoma    | invalid_key_strict  | 80630    |
| delaware    | record_type         | 3959     |
| delaware    | merchant_masked     | 0        |
| delaware    | no_code             | 669489   |
| delaware    | linked_needs_review | 0        |
| delaware    | invalid_key_strict  | 8414     |
| sunderland  | record_type         | 91       |
| sunderland  | merchant_masked     | 0        |
| sunderland  | no_code             | 41453    |
| sunderland  | linked_needs_review | 0        |
| sunderland  | invalid_key_strict  | 223      |

## Bölmeler

| role    | unique_texts   | unique_key_loose   | rows    | classes   | model_records   |
|---------|----------------|--------------------|---------|-----------|-----------------|
| test_de | 8227           | 8205               | 175961  | 176       | 8667            |
| test_ok | 16846          | 16809              | 601499  | 232       | 18293           |
| test_su | 294            | 294                | 3596    | 88        | 312             |
| train   | 134039         | 133682             | 3935335 | 253       | 145445          |
| unused  | 76439          | 76300              | 1603299 | 222       | 80356           |
| val     | 16673          | 16673              | 333069  | 228       | 18072           |

## Eğitim

| sample   | selected_c   | train_records   | train_unique_texts   | dropped_train_records   | dropped_train_rows   | dropped_train_codes   | class_count   | feature_count   | elapsed_seconds   |
|----------|--------------|-----------------|----------------------|-------------------------|----------------------|-----------------------|---------------|-----------------|-------------------|
|          | 0.300000     | 145398          | 134002               | 47                      | 101                  | 20                    | 233           | 208225          | 295.204107        |

## Ana sonuçlar

| test_set   | model           | acc_rows [95% GA]             | acc_merchant   | macro_f1 [95% GA]             | macro_f1_n_classes   | top3_rows   | group_acc_rows   | oov_rate_rows   | ceiling_rows   | acc_over_ceiling   |
|------------|-----------------|-------------------------------|----------------|-------------------------------|----------------------|-------------|------------------|-----------------|----------------|--------------------|
| test_ok    | majority        | 0.010291 [0.005117, 0.022047] | 0.083646       | 0.000882 [0.000843, 0.000921] | 175.000000           | 0.067397    | 0.125112         | 0.000015        | 0.934186       | 0.011016           |
| test_ok    | tfidf_linearsvc | 0.586463 [0.491576, 0.682286] | 0.429829       | 0.365326 [0.344765, 0.373202] | 175.000000           | 0.805315    | 0.748914         | 0.000015        | 0.934186       | 0.627780           |
| test_de    | majority        | 0.073499 [0.042348, 0.127819] | 0.350152       | 0.004322 [0.004231, 0.004419] | 120.000000           | 0.088184    | 0.174743         | 0.005223        | 0.825643       | 0.089021           |
| test_de    | tfidf_linearsvc | 0.618205 [0.484954, 0.689674] | 0.463742       | 0.298657 [0.274142, 0.309895] | 120.000000           | 0.767119    | 0.777536         | 0.005223        | 0.825643       | 0.748756           |
| test_su    | majority        | 0.012792 [0.005413, 0.039042] | 0.095238       | 0.010870 [0.007588, 0.014307] | 16.000000            | 0.098443    | 0.080089         | 0.010011        | 0.988320       | 0.012943           |
| test_su    | tfidf_linearsvc | 0.350389 [0.059501, 0.717415] | 0.343537       | 0.409956 [0.338162, 0.464407] | 16.000000            | 0.495273    | 0.420745         | 0.010011        | 0.988320       | 0.354530           |

## Hata ayrıştırması

| test_set   | model           | in_vocab_error_rows   | same_group_error_share   | diff_group_error_share   |
|------------|-----------------|-----------------------|--------------------------|--------------------------|
| test_ok    | majority        | 595300.000000         | 0.116017                 | 0.883983                 |
| test_ok    | tfidf_linearsvc | 248733.000000         | 0.543989                 | 0.456011                 |
| test_de    | majority        | 162109.000000         | 0.109661                 | 0.890339                 |
| test_de    | tfidf_linearsvc | 66262.000000          | 0.422369                 | 0.577631                 |
| test_su    | majority        | 3514.000000           | 0.067729                 | 0.932271                 |
| test_su    | tfidf_linearsvc | 2300.000000           | 0.120870                 | 0.879130                 |

## Etiket kökeni

| test_set   | model           | label_origin   | n_rows        | acc_rows   |
|------------|-----------------|----------------|---------------|------------|
| test_ok    | majority        | original       | 0.000000      |            |
| test_ok    | majority        | linked         | 601499.000000 | 0.010291   |
| test_ok    | tfidf_linearsvc | original       | 0.000000      |            |
| test_ok    | tfidf_linearsvc | linked         | 601499.000000 | 0.586463   |
| test_de    | majority        | original       | 0.000000      |            |
| test_de    | majority        | linked         | 175961.000000 | 0.073499   |
| test_de    | tfidf_linearsvc | original       | 0.000000      |            |
| test_de    | tfidf_linearsvc | linked         | 175961.000000 | 0.618205   |
| test_su    | majority        | original       | 0.000000      |            |
| test_su    | majority        | linked         | 3596.000000   | 0.012792   |
| test_su    | tfidf_linearsvc | original       | 0.000000      |            |
| test_su    | tfidf_linearsvc | linked         | 3596.000000   | 0.350389   |

## Oklahoma kapsamı

| test_set   | model           | seen_in_ok   | n_rows        | acc_rows   |
|------------|-----------------|--------------|---------------|------------|
| test_de    | majority        | True         | 107516.000000 | 0.019671   |
| test_de    | majority        | False        | 68445.000000  | 0.158054   |
| test_de    | tfidf_linearsvc | True         | 107516.000000 | 0.675230   |
| test_de    | tfidf_linearsvc | False        | 68445.000000  | 0.528629   |
| test_su    | majority        | True         | 1417.000000   | 0.002117   |
| test_su    | majority        | False        | 2179.000000   | 0.019734   |
| test_su    | tfidf_linearsvc | True         | 1417.000000   | 0.717713   |
| test_su    | tfidf_linearsvc | False        | 2179.000000   | 0.111519   |

## Sızıntı kontrolü

| protocol                 | acc_rows   | acc_merchant   | sample_rows   | train_records   | test_records   | test_rows     | test_rows_seen_in_train_share   | selected_c   | class_count   | convergence_warning_count   |
|--------------------------|------------|----------------|---------------|-----------------|----------------|---------------|---------------------------------|--------------|---------------|-----------------------------|
| random_raw_row_split     | 0.927775   |                | 1000000       | 799868          | 200000.000000  | 200000.000000 | 0.952855                        | 0.300000     | 222           | 47                          |
| merchant_grouped_test_ok | 0.586463   | 0.429829       |               |                 | 18291.000000   | 601499.000000 | 0.000000                        | 0.300000     | 233           |                             |

## C araması

| C        | val_macro_f1   | val_records   |
|----------|----------------|---------------|
| 0.100000 | 0.285983       | 18072         |
| 0.300000 | 0.301882       | 18072         |
| 1.000000 | 0.289390       | 18072         |

## En büyük hatalar: test_ok / majority

| text                    | label_code   | true_description                                                                                                                     | pred_code   | pred_description              | n_rows   | same_group   |
|-------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------|-------------|-------------------------------|----------|--------------|
| wm supercenter          | 5411         | Grocery Stores and Supermarkets                                                                                                      | 5812        | Eating Places and Restaurants | 95374    | False        |
| wal-mart                | 5411         | Grocery Stores and Supermarkets                                                                                                      | 5812        | Eating Places and Restaurants | 91776    | False        |
| united                  | 4511         | Airlines and Air Carriers (Not Elsewhere                                                                                             | 5812        | Eating Places and Restaurants | 36908    | False        |
| officemax ct in         | 5943         | Stationery Stores, Office and School Supply Stores                                                                                   | 5812        | Eating Places and Restaurants | 23294    | True         |
| walmart com             | 5310         | Discount Stores                                                                                                                      | 5812        | Eating Places and Restaurants | 18751    | False        |
| south central industrie | 5399         | Miscellaneous General Merchandise                                                                                                    | 5812        | Eating Places and Restaurants | 11351    | False        |
| best buy                | 5732         | Electronics Stores                                                                                                                   | 5812        | Eating Places and Restaurants | 11074    | True         |
| quill corporation       | 5111         | Stationery, Office Supplies, Printing and Writing Paper                                                                              | 5812        | Eating Places and Restaurants | 9568     | False        |
| homeland                | 5411         | Grocery Stores and Supermarkets                                                                                                      | 5812        | Eating Places and Restaurants | 9270     | False        |
| at t data               | 4814         | Telecommunication Services, including Local and Long-Distance Calls, Credit Card Calls, Calls Through Use of Magnetic-Stripe-Reading | 5812        | Eating Places and Restaurants | 8648     | False        |
| workquest-oklahoma      | 5964         | Direct Marketing – Catalog Merchant                                                                                                  | 5812        | Eating Places and Restaurants | 8318     | False        |
| restaurant depot        | 5411         | Grocery Stores and Supermarkets                                                                                                      | 5812        | Eating Places and Restaurants | 7393     | False        |
| ok dept of career tech  | 9399         | Government Services (Not Elsewhere Classified)                                                                                       | 5812        | Eating Places and Restaurants | 6337     | False        |
| holiday inn exp suites  | 7011         | Lodging – Hotels, Motels, Resorts, Central                                                                                           | 5812        | Eating Places and Restaurants | 6089     | False        |
| at t bill payment       | 4814         | Telecommunication Services, including Local and Long-Distance Calls, Credit Card Calls, Calls Through Use of Magnetic-Stripe-Reading | 5812        | Eating Places and Restaurants | 5287     | False        |

## En büyük hatalar: test_ok / tfidf_linearsvc

| text                      | label_code   | true_description                                                                                                                     | pred_code   | pred_description                                      | n_rows   | same_group   |
|---------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------|-------------|-------------------------------------------------------|----------|--------------|
| wal-mart                  | 5411         | Grocery Stores and Supermarkets                                                                                                      | 5310        | Discount Stores                                       | 91776    | True         |
| south central industrie   | 5399         | Miscellaneous General Merchandise                                                                                                    | 5199        | Nondurable Goods (Not Elsewhere Classified)           | 11351    | False        |
| at t data                 | 4814         | Telecommunication Services, including Local and Long-Distance Calls, Credit Card Calls, Calls Through Use of Magnetic-Stripe-Reading | 4812        | Telecommunication Equipment and Telephone Sales       | 8648     | True         |
| workquest-oklahoma        | 5964         | Direct Marketing – Catalog Merchant                                                                                                  | 5691        | Men’s and Women’s Clothing Stores                     | 8318     | False        |
| restaurant depot          | 5411         | Grocery Stores and Supermarkets                                                                                                      | 5812        | Eating Places and Restaurants                         | 7393     | False        |
| cellsignal com            | 5122         | Drugs, Drug Proprietaries, and Druggist Sundries                                                                                     | 5734        | Computer Software Stores                              | 2944     | False        |
| okapp                     | 8398         | Charitable Social Service Organizations                                                                                              | 8699        | Membership Organizations (Not Elsewhere               | 2799     | True         |
| the webstaurant store inc | 5099         | Durable Goods (Not Elsewhere Classified)                                                                                             | 5046        | Commercial Equipment (Not Elsewhere Classified)       | 2707     | True         |
| sigma-aldrich inc st lo   | 5085         | Industrial Supplies (Not Elsewhere Classified)                                                                                       | 7392        | Management, Consulting, and Public Relations Services | 2507     | False        |
| walker stamp seal         | 5099         | Durable Goods (Not Elsewhere Classified)                                                                                             | 5999        | Miscellaneous and Specialty Retail Shops              | 2324     | False        |
| govcnctn                  | 5045         | Computers and Computer Peripheral Equipment and Software                                                                             | 7399        | Business Services (Not Elsewhere Classified)          | 2237     | False        |
| buy for less              | 5411         | Grocery Stores and Supermarkets                                                                                                      | 8999        | Professional Services (Not Elsewhere Classified)      | 1992     | False        |
| red carpet charters       | 4121         | Taxicabs and Limousines                                                                                                              | 4789        | Transportation Services (Not Elsewhere                | 1869     | True         |
| heatwave supply inc       | 5074         | Plumbing and Heating Equipment and Supplies                                                                                          | 1711        | Heating, Plumbing, and Air Conditioning Contractors   | 1651     | False        |
| heritage food service gro | 5046         | Commercial Equipment (Not Elsewhere Classified)                                                                                      | 5812        | Eating Places and Restaurants                         | 1597     | False        |

## En büyük hatalar: test_de / majority

| text                    | label_code   | true_description                                   | pred_code   | pred_description              | n_rows   | same_group   |
|-------------------------|--------------|----------------------------------------------------|-------------|-------------------------------|----------|--------------|
| walmart com             | 5310         | Discount Stores                                    | 5812        | Eating Places and Restaurants | 18478    | False        |
| wal-mart                | 5411         | Grocery Stores and Supermarkets                    | 5812        | Eating Places and Restaurants | 13175    | False        |
| wm supercenter          | 5411         | Grocery Stores and Supermarkets                    | 5812        | Eating Places and Restaurants | 11424    | False        |
| state janitoral supply  | 7349         | Cleaning, Maintenance, and Janitorial Services     | 5812        | Eating Places and Restaurants | 10996    | False        |
| united electric new cas | 1799         | Special Trade Contractors (Not Elsewhere           | 5812        | Eating Places and Restaurants | 4588     | False        |
| fleetpride inc          | 5533         | Automotive Parts and Accessories Stores            | 5812        | Eating Places and Restaurants | 4309     | False        |
| wal-mart                | 5310         | Discount Stores                                    | 5812        | Eating Places and Restaurants | 3543     | False        |
| fleetpride bills online | 5533         | Automotive Parts and Accessories Stores            | 5812        | Eating Places and Restaurants | 3318     | False        |
| united electric dover   | 5065         | Electrical Parts and Equipment                     | 5812        | Eating Places and Restaurants | 3112     | False        |
| officemax ct in         | 5943         | Stationery Stores, Office and School Supply Stores | 5812        | Eating Places and Restaurants | 2952     | True         |
| napa store              | 5533         | Automotive Parts and Accessories Stores            | 5812        | Eating Places and Restaurants | 2484     | False        |
| safeway                 | 5411         | Grocery Stores and Supermarkets                    | 5812        | Eating Places and Restaurants | 2433     | False        |
| samsclub                | 5300         | Wholesale Clubs                                    | 5812        | Eating Places and Restaurants | 2393     | False        |
| millman napa auto part  | 5533         | Automotive Parts and Accessories Stores            | 5812        | Eating Places and Restaurants | 2269     | False        |
| giant                   | 5411         | Grocery Stores and Supermarkets                    | 5812        | Eating Places and Restaurants | 2112     | False        |

## En büyük hatalar: test_de / tfidf_linearsvc

| text                      | label_code   | true_description                               | pred_code   | pred_description                                                                 | n_rows   | same_group   |
|---------------------------|--------------|------------------------------------------------|-------------|----------------------------------------------------------------------------------|----------|--------------|
| wal-mart                  | 5411         | Grocery Stores and Supermarkets                | 5310        | Discount Stores                                                                  | 13175    | True         |
| united electric new cas   | 1799         | Special Trade Contractors (Not Elsewhere       | 5065        | Electrical Parts and Equipment                                                   | 4588     | False        |
| united electric dover     | 5065         | Electrical Parts and Equipment                 | 1731        | Electrical Contractors                                                           | 3112     | False        |
| the webstaurant store inc | 5099         | Durable Goods (Not Elsewhere Classified)       | 5046        | Commercial Equipment (Not Elsewhere Classified)                                  | 1551     | True         |
| restaurant depot          | 5411         | Grocery Stores and Supermarkets                | 5812        | Eating Places and Restaurants                                                    | 1424     | False        |
| mcdonald safety equipment | 5085         | Industrial Supplies (Not Elsewhere Classified) | 5046        | Commercial Equipment (Not Elsewhere Classified)                                  | 940      | True         |
| angersteins-wilmington    | 5211         | Lumber and Building Materials Stores           | 7399        | Business Services (Not Elsewhere Classified)                                     | 807      | False        |
| kent-sussex indust        | 8398         | Charitable Social Service Organizations        | 8999        | Professional Services (Not Elsewhere Classified)                                 | 711      | True         |
| pasco electric millsbro   | 5533         | Automotive Parts and Accessories Stores        | 1731        | Electrical Contractors                                                           | 604      | False        |
| main st market            | 5399         | Miscellaneous General Merchandise              | 5411        | Grocery Stores and Supermarkets                                                  | 538      | True         |
| de school conferences     | 9399         | Government Services (Not Elsewhere Classified) | 8299        | Schools and Educational Services (Not Elsewhere                                  | 485      | False        |
| bayshore ford truck       | 7538         | Automotive Service Shops (Non-Dealer)          | 5511        | Car and Truck Dealers (New and Used) Sales, Service, Repairs, Parts, and Leasing | 474      | False        |
| indeed                    | 5969         | Direct Marketing – Other Direct Marketers (Not | 7311        | Advertising Services                                                             | 465      | False        |
| bi inc ap                 | 5099         | Durable Goods (Not Elsewhere Classified)       | 5039        | Construction Materials (Not Elsewhere Classified)                                | 436      | True         |
| sherwin-williams          | 5231         | Glass, Paint, and Wallpaper Stores             | 5039        | Construction Materials (Not Elsewhere Classified)                                | 422      | False        |

## En büyük hatalar: test_su / majority

| text                   | label_code   | true_description                                   | pred_code   | pred_description              | n_rows   | same_group   |
|------------------------|--------------|----------------------------------------------------|-------------|-------------------------------|----------|--------------|
| argos ltd              | 5310         | Discount Stores                                    | 5812        | Eating Places and Restaurants | 1071     | False        |
| campaignmonitor        | 7399         | Business Services (Not Elsewhere Classified)       | 5812        | Eating Places and Restaurants | 984      | False        |
| premier                | 7011         | Lodging – Hotels, Motels, Resorts, Central         | 5812        | Eating Places and Restaurants | 277      | False        |
| asda george com leeds  | 5411         | Grocery Stores and Supermarkets                    | 5812        | Eating Places and Restaurants | 249      | False        |
| newcastle park and fly | 7523         | Parking Lots, Parking Meters and Garages           | 5812        | Eating Places and Restaurants | 101      | False        |
| asda stores ltd        | 5411         | Grocery Stores and Supermarkets                    | 5812        | Eating Places and Restaurants | 50       | False        |
| flickr pro monthly     | 7399         | Business Services (Not Elsewhere Classified)       | 5812        | Eating Places and Restaurants | 47       | False        |
| smythstoys com         | 5945         | Hobby, Toy, and Game Shops                         | 5812        | Eating Places and Restaurants | 33       | True         |
| ryman                  | 5943         | Stationery Stores, Office and School Supply Stores | 5812        | Eating Places and Restaurants | 29       | True         |
| b d plastics ltd       | 5074         | Plumbing and Heating Equipment and Supplies        | 5812        | Eating Places and Restaurants | 25       | False        |
| hse - bsr              | 9399         | Government Services (Not Elsewhere Classified)     | 5812        | Eating Places and Restaurants | 20       | False        |
| the range              | 5311         | Department Stores                                  | 5812        | Eating Places and Restaurants | 17       | False        |
| the range              | 5719         | Miscellaneous Home Furnishing Specialty Stores     | 5812        | Eating Places and Restaurants | 17       | True         |
| crushftp com           | 5734         | Computer Software Stores                           | 5812        | Eating Places and Restaurants | 16       | True         |
| asda direct web        | 5411         | Grocery Stores and Supermarkets                    | 5812        | Eating Places and Restaurants | 15       | False        |

## En büyük hatalar: test_su / tfidf_linearsvc

| text                   | label_code   | true_description                                   | pred_code   | pred_description                                                  | n_rows   | same_group   |
|------------------------|--------------|----------------------------------------------------|-------------|-------------------------------------------------------------------|----------|--------------|
| argos ltd              | 5310         | Discount Stores                                    | 5812        | Eating Places and Restaurants                                     | 1071     | False        |
| premier                | 7011         | Lodging – Hotels, Motels, Resorts, Central         | 4812        | Telecommunication Equipment and Telephone Sales                   | 277      | False        |
| asda george com leeds  | 5411         | Grocery Stores and Supermarkets                    | 5099        | Durable Goods (Not Elsewhere Classified)                          | 249      | False        |
| newcastle park and fly | 7523         | Parking Lots, Parking Meters and Garages           | 7311        | Advertising Services                                              | 101      | True         |
| asda stores ltd        | 5411         | Grocery Stores and Supermarkets                    | 5399        | Miscellaneous General Merchandise                                 | 50       | True         |
| flickr pro monthly     | 7399         | Business Services (Not Elsewhere Classified)       | 7221        | Photographic Studios                                              | 47       | False        |
| ryman                  | 5943         | Stationery Stores, Office and School Supply Stores | 5691        | Men’s and Women’s Clothing Stores                                 | 29       | False        |
| b d plastics ltd       | 5074         | Plumbing and Heating Equipment and Supplies        | 5099        | Durable Goods (Not Elsewhere Classified)                          | 25       | True         |
| hse - bsr              | 9399         | Government Services (Not Elsewhere Classified)     | 8299        | Schools and Educational Services (Not Elsewhere                   | 20       | False        |
| the range              | 5311         | Department Stores                                  | 5812        | Eating Places and Restaurants                                     | 17       | False        |
| the range              | 5719         | Miscellaneous Home Furnishing Specialty Stores     | 5812        | Eating Places and Restaurants                                     | 17       | True         |
| crushftp com           | 5734         | Computer Software Stores                           | 5941        | Sporting Goods Stores                                             | 16       | True         |
| asda direct web        | 5411         | Grocery Stores and Supermarkets                    | 8699        | Membership Organizations (Not Elsewhere                           | 15       | False        |
| gro hm passport office | 9399         | Government Services (Not Elsewhere Classified)     | 7399        | Business Services (Not Elsewhere Classified)                      | 15       | False        |
| lul ticket machine     | 4112         | Passenger Railways                                 | 8220        | Colleges, Universities, Professional Schools, and Junior Colleges | 15       | False        |
