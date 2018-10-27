def remove_empty_rows(*input_cols):
    indices = list(range(0, num_images, CLUSTER_SIZE))
    output_cols = []
    for col in input_cols:
        output_cols.append(db.streams.Gather(col, indices))
    return output_cols
