from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import joblib
import pandas as pd

# =========================================================
# APP SETUP
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

# =========================================================
# DATA PATHS
# =========================================================

FEATURE_DATA_PATH = BASE_DIR / "data" / "processed" / "feature_data.csv"
FRAUD_DATA_PATH = BASE_DIR / "data" / "processed" / "fraud_data.csv"
RECOMMENDATION_DATA_PATH = BASE_DIR / "data" / "processed" / "recommendation_data.csv"

# =========================================================
# MODEL PATHS
# =========================================================

CHURN_MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
CLV_MODEL_PATH = BASE_DIR / "models" / "clv_model.pkl"
FRAUD_MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

# =========================================================
# LOAD DATA
# =========================================================

feature_data = pd.read_csv(FEATURE_DATA_PATH)
fraud_data = pd.read_csv(FRAUD_DATA_PATH)
recommendation_data = pd.read_csv(RECOMMENDATION_DATA_PATH)

# =========================================================
# CLEAN ID COLUMNS
# =========================================================

def clean_id_column(df, column):

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )


clean_id_column(feature_data, "Customer_ID")

clean_id_column(fraud_data, "Transaction_ID")
clean_id_column(fraud_data, "Customer_ID")

clean_id_column(recommendation_data, "Customer_ID")


# =========================================================
# LOAD MODELS
# =========================================================

churn_model = joblib.load(CHURN_MODEL_PATH)

clv_model = joblib.load(CLV_MODEL_PATH)

fraud_model = joblib.load(FRAUD_MODEL_PATH)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_float(value):

    try:

        if pd.isna(value):
            return 0.0

        return float(value)

    except:

        return 0.0


def safe_int(value):

    try:

        if pd.isna(value):
            return 0

        return int(float(value))

    except:

        return 0


# =========================================================
# MODEL INPUT FUNCTION
# =========================================================

def make_model_input(model, values):

    if hasattr(model, "feature_names_in_"):

        features = list(model.feature_names_in_)

        row = {}

        for feature in features:

            value = values.get(feature, 0)

            if pd.isna(value):

                value = 0

            row[feature] = value

        return pd.DataFrame(
            [row],
            columns=features
        )

    return pd.DataFrame([values])


# =========================================================
# CUSTOMER FEATURES
# ONLY 3 FEATURES SHOWN ON INTERFACE
# =========================================================

def customer_insights(row):

    return {

        "total_spending":
            safe_float(
                row.get("total_spent", 0)
            ),

        "total_transactions":
            safe_int(
                row.get("total_transactions", 0)
            ),

        "recent_activity":
            safe_int(
                row.get("total_sessions", 0)
            )
    }


# =========================================================
# CUSTOMER DROPDOWN
# =========================================================

def get_customer_options():

    required_columns = [
        "Customer_ID",
        "City",
        "Age"
    ]

    available_columns = [
        col
        for col in required_columns
        if col in feature_data.columns
    ]

    df = feature_data[
        available_columns
    ].drop_duplicates(
        "Customer_ID"
    )

    options = []

    for _, row in df.iterrows():

        customer_id = str(
            row["Customer_ID"]
        ).strip()

        if not customer_id:
            continue

        details = []

        if "City" in row:

            city = str(
                row["City"]
            ).strip()

            if city and city.lower() != "nan":

                details.append(city)

        if "Age" in row:

            age = safe_int(
                row["Age"]
            )

            if age:

                details.append(
                    f"Age {age}"
                )

        if details:

            label = (
                "Customer — "
                + " — ".join(details)
            )

        else:

            label = (
                "Customer "
                + customer_id
            )

        options.append({

            "id": customer_id,

            "label": label

        })

    return sorted(
        options,
        key=lambda x: x["id"]
    )


# =========================================================
# TRANSACTION DROPDOWN
# =========================================================

def get_transaction_options():

    required_columns = [
        "Transaction_ID",
        "Purchase_Amount",
        "Brand",
        "Payment_Method"
    ]

    available_columns = [
        col
        for col in required_columns
        if col in fraud_data.columns
    ]

    df = fraud_data[
        available_columns
    ].drop_duplicates(
        "Transaction_ID"
    )

    options = []

    for _, row in df.iterrows():

        transaction_id = str(
            row["Transaction_ID"]
        ).strip()

        if not transaction_id:
            continue

        parts = []

        if "Purchase_Amount" in row:

            amount = safe_float(
                row["Purchase_Amount"]
            )

            parts.append(
                f"₹{amount:,.0f}"
            )

        if "Brand" in row:

            brand = str(
                row["Brand"]
            ).strip()

            if brand and brand.lower() != "nan":

                parts.append(brand)

        options.append({

            "id": transaction_id,

            "label":
                "Transaction — "
                + " — ".join(parts)

        })

    return sorted(
        options,
        key=lambda x: x["id"]
    )


CUSTOMER_OPTIONS = get_customer_options()

TRANSACTION_OPTIONS = get_transaction_options()


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    total_customers = (
        feature_data[
            "Customer_ID"
        ].nunique()
    )

    total_transactions = (
        fraud_data[
            "Transaction_ID"
        ].nunique()
    )

    total_spending = safe_float(
        fraud_data[
            "Purchase_Amount"
        ].sum()
    )

    if "Fraud" in fraud_data.columns:

        fraud_rate = (
            safe_float(
                fraud_data[
                    "Fraud"
                ].mean()
            )
            * 100
        )

    else:

        fraud_rate = 0

    return render_template(

        "index.html",

        total_customers=
            total_customers,

        total_transactions=
            total_transactions,

        total_spending=
            total_spending,

        fraud_rate=
            fraud_rate,

        page="home"
    )


# =========================================================
# CUSTOMER MAIN PAGE
# =========================================================

@app.route("/customer")
def customer():

    return render_template(

        "customer.html",

        page="customer"

    )


# =========================================================
# CUSTOMER RISK
# =========================================================

@app.route(
    "/customer-risk",
    methods=["GET", "POST"]
)
def customer_risk():

    prediction = None

    insights = None

    error = None

    selected_id = ""

    if request.method == "POST":

        selected_id = (
            request.form
            .get(
                "customer_id",
                ""
            )
            .strip()
        )

        rows = feature_data[
            feature_data[
                "Customer_ID"
            ] == selected_id
        ]

        if rows.empty:

            error = (
                "Customer not found. "
                "Please choose a customer "
                "from the list."
            )

        else:

            row = rows.iloc[0]

            insights = customer_insights(
                row
            )

            values = row.to_dict()

            try:

                model_input = (
                    make_model_input(
                        churn_model,
                        values
                    )
                )

                result = int(
                    churn_model.predict(
                        model_input
                    )[0]
                )

                if result == 1:

                    prediction = {

                        "type":
                            "warning",

                        "title":
                            "May Need Attention",

                        "message":
                            "This customer may stop buying from you.",

                        "advice":
                            "Check their recent activity and consider giving helpful support or an offer."

                    }

                else:

                    prediction = {

                        "type":
                            "success",

                        "title":
                            "Likely to Stay",

                        "message":
                            "This customer is likely to continue buying from you.",

                        "advice":
                            "Keep giving this customer a good experience."

                    }

            except Exception as e:

                print(
                    "CHURN ERROR:",
                    repr(e)
                )

                error = (
                    "We could not check "
                    "this customer. "
                    "Please try another customer."
                )

    return render_template(

        "customer_risk.html",

        prediction=prediction,

        customer_insights=insights,

        error=error,

        customers=CUSTOMER_OPTIONS,

        selected_id=selected_id,

        page="customer-risk"

    )


# =========================================================
# CUSTOMER VALUE
# =========================================================

@app.route(
    "/customer-value",
    methods=["GET", "POST"]
)
def customer_value():

    prediction = None

    insights = None

    error = None

    selected_id = ""

    if request.method == "POST":

        selected_id = (
            request.form
            .get(
                "customer_id",
                ""
            )
            .strip()
        )

        rows = feature_data[
            feature_data[
                "Customer_ID"
            ] == selected_id
        ]

        if rows.empty:

            error = (
                "Customer not found. "
                "Please choose a customer "
                "from the list."
            )

        else:

            row = rows.iloc[0]

            insights = customer_insights(
                row
            )

            try:

                model_input = (
                    make_model_input(
                        clv_model,
                        row.to_dict()
                    )
                )

                result = float(
                    clv_model.predict(
                        model_input
                    )[0]
                )

                if result >= 100000:

                    title = (
                        "Very Valuable Customer"
                    )

                elif result >= 50000:

                    title = (
                        "High-Value Customer"
                    )

                elif result >= 10000:

                    title = (
                        "Growing Customer"
                    )

                else:

                    title = (
                        "Regular Customer"
                    )

                prediction = {

                    "title":
                        title,

                    "value":
                        f"₹{result:,.2f}",

                    "message":
                        "This is the estimated future value of this customer."

                }

            except Exception as e:

                print(
                    "CLV ERROR:",
                    repr(e)
                )

                error = (
                    "We could not estimate "
                    "this customer's value."
                )

    return render_template(

        "customer_value.html",

        prediction=prediction,

        customer_insights=insights,

        error=error,

        customers=CUSTOMER_OPTIONS,

        selected_id=selected_id,

        page="customer-value"

    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

@app.route(
    "/recommendations",
    methods=["GET", "POST"]
)
def recommendations():

    suggestions = None

    insights = None

    error = None

    selected_id = ""

    if request.method == "POST":

        selected_id = (
            request.form
            .get(
                "customer_id",
                ""
            )
            .strip()
        )

        rows = feature_data[
            feature_data[
                "Customer_ID"
            ] == selected_id
        ]

        if rows.empty:

            error = (
                "Customer not found. "
                "Please choose a customer "
                "from the list."
            )

        else:

            row = rows.iloc[0]

            insights = customer_insights(
                row
            )

            history = recommendation_data[
                recommendation_data[
                    "Customer_ID"
                ] == selected_id
            ]

            if history.empty:

                error = (
                    "There is not enough activity "
                    "to make suggestions."
                )

            else:

                used = set(
                    history.loc[
                        history[
                            "Interaction"
                        ] == 1,
                        "Brand"
                    ]
                    .dropna()
                    .astype(str)
                )

                available = (
                    recommendation_data[
                        ~recommendation_data[
                            "Brand"
                        ].astype(str)
                        .isin(used)
                    ]
                )

                suggestions = (
                    available[
                        "Brand"
                    ]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(5)
                    .index
                    .tolist()
                )

                if not suggestions:

                    suggestions = list(
                        used
                    )[:5]

                if not suggestions:

                    error = (
                        "No useful suggestions "
                        "were found."
                    )

    return render_template(

        "recommendations.html",

        suggestions=suggestions,

        customer_insights=insights,

        error=error,

        customers=CUSTOMER_OPTIONS,

        selected_id=selected_id,

        page="recommendations"

    )


# =========================================================
# TRANSACTION MAIN PAGE
# =========================================================

@app.route("/transaction")
def transaction():

    return render_template(

        "transaction.html",

        page="transaction"

    )


# =========================================================
# TRANSACTION SAFETY
# =========================================================

@app.route(
    "/transaction-safety",
    methods=["GET", "POST"]
)
def transaction_safety():

    prediction = None

    insights = None

    error = None

    selected_id = ""

    if request.method == "POST":

        selected_id = (
            request.form
            .get(
                "transaction_id",
                ""
            )
            .strip()
        )

        rows = fraud_data[
            fraud_data[
                "Transaction_ID"
            ] == selected_id
        ]

        if rows.empty:

            error = (
                "Transaction not found. "
                "Please choose a transaction "
                "from the list."
            )

        else:

            row = rows.iloc[0]

            insights = {

                "amount":
                    safe_float(
                        row.get(
                            "Purchase_Amount",
                            0
                        )
                    ),

                "payment":
                    str(
                        row.get(
                            "Payment_Method",
                            "Not available"
                        )
                    ),

                "brand":
                    str(
                        row.get(
                            "Brand",
                            "Not available"
                        )
                    )

            }

            try:

                model_input = (
                    make_model_input(
                        fraud_model,
                        row.to_dict()
                    )
                )

                result = int(
                    fraud_model.predict(
                        model_input
                    )[0]
                )

                if result == 1:

                    prediction = {

                        "type":
                            "danger",

                        "title":
                            "Needs Attention",

                        "message":
                            "This transaction looks unusual.",

                        "advice":
                            "Review the transaction before approving it."

                    }

                else:

                    prediction = {

                        "type":
                            "success",

                        "title":
                            "Looks Safe",

                        "message":
                            "No strong warning signs were found.",

                        "advice":
                            "There is no immediate reason for concern."

                    }

            except Exception as e:

                print(
                    "FRAUD ERROR:",
                    repr(e)
                )

                error = (
                    "We could not check "
                    "this transaction."
                )

    return render_template(

        "transaction_safety.html",

        prediction=prediction,

        transaction_insights=insights,

        error=error,

        transactions=
            TRANSACTION_OPTIONS,

        selected_id=selected_id,

        page="transaction-safety"

    )


# =========================================================
# OLD FRAUD URL
# =========================================================

@app.route(
    "/fraud",
    methods=["GET", "POST"]
)
def fraud_redirect():

    return redirect(
        url_for(
            "transaction_safety"
        )
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

@app.route("/insights")
def insights():

    total_customers = (
        feature_data[
            "Customer_ID"
        ].nunique()
    )

    total_transactions = (
        fraud_data[
            "Transaction_ID"
        ].nunique()
    )

    total_spending = safe_float(
        fraud_data[
            "Purchase_Amount"
        ].sum()
    )

    if "Fraud" in fraud_data.columns:

        fraud_rate = (
            safe_float(
                fraud_data[
                    "Fraud"
                ].mean()
            )
            * 100
        )

    else:

        fraud_rate = 0

    return render_template(

        "insights.html",

        total_customers=
            total_customers,

        total_transactions=
            total_transactions,

        total_spending=
            total_spending,

        fraud_rate=
            fraud_rate,

        page="insights"

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )