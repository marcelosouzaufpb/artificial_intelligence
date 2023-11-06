import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def create_dataframe():
    df_backup = pd.DataFrame([
        ['yellow', 'XL', 79.90, 'class1'],
        ['blue', 'M', 49.50, 'class2'],
        ['black', 'S', 54.30, 'class3'],
        ['black', 'XS', 65.00, 'class2'],
        ['white', 'M', 69.90, 'class3'],
        ['gray', 'L', 75.90, 'class2'],
        ['brown', 'XL', 59.80, 'class3']
    ])

    df_backup.columns = ['color', 'size', 'price', 'class']
    df = df_backup.copy()
    return df


def explore_unique_values(df):
    categorical_cols = ['color', 'size', 'class']
    unique_counts = df[categorical_cols].apply(lambda x: x.nunique(), axis=0)
    print("Number of unique values for each categorical column:")
    print(unique_counts)

    for cat in categorical_cols:
        unique_values = df[cat].unique()
        print(f"Unique values for {cat}: {unique_values}")


def map_size_to_integers(df):
    size_map = {'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4}
    df['size'] = df['size'].map(size_map)
    return df


def reverse_size_mapping(df):
    size_map = {'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4}
    inv_size_map = {v: k for k, v in size_map.items()}
    original_sorted = sorted(size_map.items(), key=lambda x: x[1])

    print("Original size mapping:", original_sorted)
    print("Reversed size mapping:", inv_size_map)


def encode_categorical_variables(df):
    encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']):
        df[col] = encoder.fit_transform(df[col])
    return df


def decoded_categorical_variables(df):
    color_map = {0: 'black', 1: 'blue', 2: 'brown', 3: 'gray', 4: 'white', 5: 'yellow'}
    size_map = {0: 'XS', 1: 'S', 2: 'M', 3: 'L', 4: 'XL'}
    class_map = {0: 'class1', 1: 'class2', 2: 'class3'}

    df['color'] = df['color'].map(color_map)
    df['size'] = df['size'].map(size_map)
    df['class'] = df['class'].map(class_map)

    return df


if __name__ == "__main__":
    df = create_dataframe()
    print(df, '\n')

    explore_unique_values(df)

    df = map_size_to_integers(df)
    print(df, '\n')

    reverse_size_mapping(df)

    df_encoded = encode_categorical_variables(df)
    print(df_encoded, '\n')

    df_decoded = decoded_categorical_variables(df)
    print(df_decoded, '\n')


def encode_labels(df, class_name='class'):
    class_label_encoder = LabelEncoder()
    y = class_label_encoder.fit_transform(df[class_name].values)
    return y, class_label_encoder


def decode_labels(encoded_labels, label_encoder):
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    return decoded_labels


def encode_labels_ordinal(df, class_name='class', categories_data=[]):
    ordinal_encoder = OrdinalEncoder(categories=categories_data)
    df[class_name] = ordinal_encoder.fit_transform(df[class_name].values.reshape(-1, 1))
    return df


if __name__ == "__main__":
    df = create_dataframe()
    print(df, '\n')

    y, class_label_encoder = encode_labels(df, 'class')
    print("Encoded class labels:")
    print(y, '\n')

    x = decode_labels(y, class_label_encoder)
    print("Decoded class labels:")
    print(x, '\n')

    z, color_label_encoder = encode_labels(df, 'color')
    print("Encoded color labels:")
    print(z, '\n')

    df = create_dataframe()

    categories = [['XS', 'S', 'M', 'L', 'XL']]
    w = encode_labels_ordinal(df, 'size', categories)
    print("Encode labels ordinal:")
    print(w, '\n')
