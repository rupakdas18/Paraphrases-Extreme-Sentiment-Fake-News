
def print(train_df, test_df, valid_df):
    print("Shape before removing NAN values")
    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)


    train_df = train_df.dropna(how='any')
    test_df = test_df.dropna(how='any')
    valid_df = valid_df.dropna(how='any')

    print("Shape after removing NAN values")

    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)

