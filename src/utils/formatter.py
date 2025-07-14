import streamlit as st

# def parse_dollar_input(label, default):
#     val_str = st.text_input(label, value=f"{default:,}")
#     try:
#         return int(val_str.replace(",", "").replace("$", ""))
#     except ValueError:
#         st.warning(f"Please enter a valid number for '{label}'.")
#         return 0

# current_assets = parse_dollar_input("Current Portfolio Value ($)", 500_000)


def parse_dollar_input(label: str, default: int) -> int:
    """
    Text input that displays a comma-formatted version of the value below the field.
    """
    val_str = st.text_input(label, value=f"{default:,}")
    try:
        parsed_value = int(val_str.replace(",", "").replace("$", "").strip())
        st.caption(f"➡️ Interpreted as: **${parsed_value:,.0f}**")
        return parsed_value
    except ValueError:
        st.warning(f"Please enter a valid number for '{label}'.")
        return 0