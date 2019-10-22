## TOPIC MODEL

### run

edit the database config first, in the file `database-config.json`

`python3 topic_modeling.py <product_id>`

### what it does?

- connect to database to get all reviews about the given `product`
- perform topic-model on those reviews
- show the magic

### example: 
```
$ python3 topic_modeling.py 2403857

Fetched 144 reviews for product_id = 2403857
MySQL connection is closed
Topics found via LDA:

Topic #0:
đôi nh ưng_ý co mi ôm_chân hơi bền đen đi la_i rô_ng k_a n_tra muô phãi hunter mua đẹp dự_kiến

Topic #1:
giày chân size mua đi đôi hơi êm đẹp ôm thoải_mái form order bitis cổ đầu hàng cm bí giao

Topic #2:
màu đôi mẩu hình đen bitis size nam đế nhẹ nữ viết cam phối hunter giá mua hàng kèm việt

Topic #3:
giày size đẹp đi êm mua đôi chân hộp hàng chất_lượng thoải_mái nhẹ giá good hãng hunter bitis ủng_hộ hơi

Topic #4:
hàng giao nhanh sản_phẩm giày êm đẹp mua hài_lòng chân chất_lượng cẩn_thận đi đóng_gói hơi thoải_mái sz bitis shipper siêu
```

The results somehow shows the very brief of all reviews.
