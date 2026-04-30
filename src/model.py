from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_engineering import FEATURE_COLS

def train_outcome_model(df):
    """""
    Train a random forest model to predict whether a team will win, draw, or lose a world cup match.
    """""
    
    #initialize the input feature 
    x = df[FEATURE_COLS]
    y = df["result_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    #Create a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

    #Train_model
    model.fit(X_train, y_train)

    #make predictions on test data
    y_pred = model.predict(X_test)

    #Print out results
    print("Outcome Model Results")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, target_names=["lose", "draw", "win"]))

    result_map = {
        0: "lose",
        1: "draw",
        2: "win"
    }

    print("\nExample Match Predictions:")
    examples = X_test.copy()
    examples["actual_result"] = y_test
    examples["predicted_result"] = y_pred

    
    examples["team_name"] = df.loc[X_test.index, "team_name"]
    examples["opponent_name"] = df.loc[X_test.index, "opponent_name"]
    examples["match_date"] = df.loc[X_test.index, "match_date"]

    for _, row in examples.head(10).iterrows():
        team = row["team_name"]
        actual = result_map[row["actual_result"]]
        predicted = result_map[row["predicted_result"]]

        print(f"{row['match_date']} | {team} vs {opponent}")
        print(f"Actual: {actual} | Predicted: {predicted}")
        print("-" * 50)


    return model