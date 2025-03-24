import pandas as pd

# 读取数据
imagebyscore_df = pd.read_csv('/home/ps/temp/model/aesthetic2/MTCL_main/code/PIAA/FlickrAES_PIAA/label/image_labeled_by_each_worker.csv')
test_workerId_df = pd.read_csv('/home/ps/temp/model/aesthetic2/MTCL_main/code/PIAA/FlickrAES_PIAA/label/test_workers_ID.csv')

# 获取 test_workerId.csv 中的 worker IDs
test_workers = test_workerId_df['ID'].tolist()

# 筛选出在 imagebyscore.csv 中由 test_workers 评价的图片对（imagePair）
filtered_df = imagebyscore_df[imagebyscore_df['worker'].isin(test_workers)]

# 使用 groupby 按照 imagePair 进行分组，统计每个 imagePair 被多少个不同的 worker 评价
image_pair_counts = filtered_df.groupby(' imagePair')['worker'].nunique()

# 找出被至少 2 个 worker 评价的图片对
image_pairs_with_multiple_workers = image_pair_counts[image_pair_counts >= 2].index.tolist()
# 按 imagePair 分组，筛选出同时被两个不同的 worker 评价的图片
grouped = filtered_df.groupby(' imagePair').filter(lambda x: len(x['worker'].unique()) >= 2)

# 打印每个 imagePair 被哪些 worker 评价以及评分
for image_pair, group in grouped.groupby(' imagePair'):
    workers = group['worker'].tolist()  # 获取所有评价该图片的 worker
    scores = group[' score'].tolist()    # 获取所有对应的评分
    print(f"图片对 '{image_pair}' 被以下 worker 评价：{workers}")
    print(f"评分分别为：{scores}")
    print("-" * 30)