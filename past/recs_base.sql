SELECT
  invoices.cust_person_id,
  invoices.date_of_day,
  -- invoices.month_id,
  -- invoices.store_id,
  (invoices.art_no * 1000000 + invoices.var_tu_key) AS product_id,
  invoices.sell_val_nsp,
  -- stores.region_desc,
  -- stores.oblast,
  stores.city
FROM
  `cf-mccuagcf-recommenders-ht.segmentation.ml_bi_invoices_tbl` AS invoices
LEFT JOIN
  `cf-mccuagcf-recommenders-ht.segmentation.ml_bi_stores_tbl` AS stores
ON
  invoices.store_id = stores.store_id
WHERE
  invoices.month_id = 202506
  AND invoices.flag_cust_target_group = "SCO"
  AND sell_val_nsp > 0
  and city = "TERNOPIL"