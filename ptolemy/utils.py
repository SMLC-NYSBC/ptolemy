def prep_state_for_csv(df):
    # df['center_x'] = df['center'].apply(lambda x: x[0])
    # df['center_y'] = df['center'].apply(lambda x: x[1])
    columns = list(df.columns)
    columns = [col for col in columns if col not in ['features']]
    df = df[columns]
    return df


def clean_vertex_columns(df):
    df['vertices'] = df.apply(lambda row: 
        [
            [row['vert_1_x'], row['vert_1_y']],
            [row['vert_2_x'], row['vert_2_y']],
            [row['vert_3_x'], row['vert_3_y']],
            [row['vert_4_x'], row['vert_4_y']]
        ]
    )

    columns = list(df.columns)
    columns = [col for col in columns if 'vert_' not in col]
    df = df[columns]
    return df