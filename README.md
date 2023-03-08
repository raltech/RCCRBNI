# Rapid Calibration of a Cellular-Resolution Bidirectional Neural Interface (RCCRBNI)

## Result
- Running Epsilon Greedy for 50000 iteration on 2022-11-28-1 dataset, then act greedy according to the produced Q
```
Number of non-zero entries: 1762

Top 10 Non-zero entries:
Row: 19838, Column: 0, Value: 5.9567670822143555
Row: 14036, Column: 0, Value: 4.095099925994873
Row: 176, Column: 0, Value: 3.76705002784729
Row: 9458, Column: 0, Value: 3.76705002784729
Row: 2738, Column: 0, Value: 3.730599880218506
Row: 15678, Column: 0, Value: 3.4951000213623047
Row: 10488, Column: 0, Value: 3.439000129699707
Row: 13700, Column: 0, Value: 3.439000129699707
Row: 14175, Column: 0, Value: 3.439000129699707
Row: 14994, Column: 0, Value: 3.439000129699707

Original dict:  (5138, 273)
New dict:  (1762, 273)
Original Span:  226
New Span:  223
Original Magnitude Cosine Similarity:  208.15777130551677
New Magnitude Cosine Similarity:  182.17056389141385
```

- Running Epsilon Greedy for 50000 iteration on 2022-11-04-2 dataset, then act greedy according to the produced Q
```
Number of non-zero entries: 1525

Top 10 Non-zero entries:
Row: 894, Column: 0, Value: 6.2386603355407715
Row: 2063, Column: 0, Value: 3.730599880218506
Row: 189, Column: 0, Value: 3.439000129699707
Row: 1309, Column: 0, Value: 3.439000129699707
Row: 2059, Column: 0, Value: 3.439000129699707
Row: 4060, Column: 0, Value: 3.439000129699707
Row: 4771, Column: 0, Value: 3.439000129699707
Row: 15253, Column: 0, Value: 3.439000129699707
Row: 15260, Column: 0, Value: 3.439000129699707
Row: 15552, Column: 0, Value: 3.439000129699707

Original dict:  (4702, 147)
New dict:  (1525, 147)
Original Span:  111
New Span:  110
Original Magnitude Cosine Similarity:  107.67223574506016
New Magnitude Cosine Similarity:  97.75214248890961
```
