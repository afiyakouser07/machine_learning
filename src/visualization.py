# visualization.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Visualizer:

    def __init__(self, dataframe):
        self.df = dataframe

    # -------------------------------------------------
    # KPI Calculations
    # -------------------------------------------------

    def get_kpis(self):

        return {
            "total_users": len(self.df),

            "avg_screen_time":
                round(
                    self.df[
                        "Screen On Time (hours/day)"
                    ].mean(),
                    2
                ),

            "avg_battery_drain":
                round(
                    self.df[
                        "Battery Drain (mAh/day)"
                    ].mean(),
                    2
                ),

            "avg_data_usage":
                round(
                    self.df[
                        "Data Usage (MB/day)"
                    ].mean(),
                    2
                )
        }

    # -------------------------------------------------
    # Age Distribution
    # -------------------------------------------------

    def age_distribution(self):

        fig = px.histogram(
            self.df,
            x="Age",
            nbins=20,
            title="Age Distribution",
            template="plotly_white"
        )

        return fig

    # -------------------------------------------------
    # Gender Distribution
    # -------------------------------------------------

    def gender_distribution(self):

        fig = px.pie(
            self.df,
            names="Gender",
            title="Gender Distribution"
        )

        return fig

    # -------------------------------------------------
    # Operating System Distribution
    # -------------------------------------------------

    def os_distribution(self):

        fig = px.bar(
            self.df[
                "Operating System"
            ].value_counts().reset_index(),
            x="Operating System",
            y="count",
            title="Operating System Usage",
            template="plotly_white"
        )

        return fig

    # -------------------------------------------------
    # Device Model Analysis
    # -------------------------------------------------

    def device_model_analysis(self):

        device_count = (
            self.df["Device Model"]
            .value_counts()
            .reset_index()
        )

        device_count.columns = [
            "Device Model",
            "Count"
        ]

        fig = px.bar(
            device_count,
            x="Device Model",
            y="Count",
            title="Device Popularity",
            template="plotly_white"
        )

        return fig

    # -------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------

    def correlation_heatmap(self):

        numeric_df = self.df.select_dtypes(
            include="number"
        )

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix"
        )

        return fig

    # -------------------------------------------------
    # Battery Drain vs Screen Time
    # -------------------------------------------------

    def battery_vs_screen_time(self):

        fig = px.scatter(
            self.df,
            x="Screen On Time (hours/day)",
            y="Battery Drain (mAh/day)",
            color="User Behavior Class",
            title="Battery Drain vs Screen Time",
            template="plotly_white"
        )

        return fig

    # -------------------------------------------------
    # App Usage vs Data Usage
    # -------------------------------------------------

    def app_usage_vs_data_usage(self):

        fig = px.scatter(
            self.df,
            x="App Usage Time (min/day)",
            y="Data Usage (MB/day)",
            color="User Behavior Class",
            title="App Usage vs Data Usage",
            template="plotly_white"
        )

        return fig

    # -------------------------------------------------
    # Age vs App Usage
    # -------------------------------------------------

    def age_vs_app_usage(self):

        grouped = (
            self.df.groupby("Age")
            ["App Usage Time (min/day)"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            grouped,
            x="Age",
            y="App Usage Time (min/day)",
            title="Age vs Average App Usage",
            markers=True
        )

        return fig

    # -------------------------------------------------
    # User Behavior Class Distribution
    # -------------------------------------------------

    def behavior_class_distribution(self):

        fig = px.bar(
            self.df[
                "User Behavior Class"
            ].value_counts().reset_index(),
            x="User Behavior Class",
            y="count",
            title="Behavior Class Distribution"
        )

        return fig

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    def feature_importance(self, model, feature_names):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance":
                model.feature_importances_

        })

        importance = (
            importance
            .sort_values(
                by="Importance",
                ascending=False
            )
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )

        return fig

    # -------------------------------------------------
    # Insights Generator
    # -------------------------------------------------

    def generate_insights(self):

        insights = []

        avg_screen = self.df[
            "Screen On Time (hours/day)"
        ].mean()

        avg_battery = self.df[
            "Battery Drain (mAh/day)"
        ].mean()

        avg_data = self.df[
            "Data Usage (MB/day)"
        ].mean()

        if avg_screen > 5:
            insights.append(
                "High average screen time detected."
            )

        if avg_battery > 1500:
            insights.append(
                "Battery consumption is relatively high."
            )

        if avg_data > 1000:
            insights.append(
                "Users consume significant mobile data."
            )

        if len(insights) == 0:
            insights.append(
                "Usage patterns appear normal."
            )

        return insights


# -------------------------------------------------
# Testing
# -------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv(
        "data/user_behavior_dataset.csv"
    )

    viz = Visualizer(df)

    print(viz.get_kpis())

    fig = viz.age_distribution()

    fig.show()
