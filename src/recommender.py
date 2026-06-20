def generate_recommendation(customer):

    recommendations = []

    if customer["MonthlyCharges"] >= 80:
        recommendations.append(
            "Offer 20% discount on long-term plans"
        )

    if customer["tenure"] < 12:
        recommendations.append(
            "Provide onboarding support and loyalty rewards"
        )

    if customer["Contract_One year"] == 0 and customer["Contract_Two year"] == 0:
        recommendations.append(
            "Promote annual subscription plans"
        )

    if customer["TechSupport_Yes"] == 1:
        recommendations.append(
            "Offer premium technical support package"
        )

    if customer["OnlineSecurity_Yes"] == 0:
        recommendations.append(
            "Offer free online security upgrade"
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Customer is stable. Maintain engagement."
        )

    return recommendations