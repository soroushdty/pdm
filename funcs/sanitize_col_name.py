def sanitize_col_name(model_id: str) -> str:
    """Sanitize model id into a safe DF column name."""
    return model_id.replace("/", "__").replace("-", "_")
