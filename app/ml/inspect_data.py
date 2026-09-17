"""Read-only inspection and contract validation for the real churn dataset."""

from typing import TextIO
import sys

import pandas as pd

from app.config import DATASET_PATH
from app.ml.data import (
    read_dataset,
    validate_dataset_columns,
    validate_dataset_domains,
)
from app.ml.diagnostics import DuplicateScopeSummary, analyze_duplicates
from app.ml.features import (
    BINARY_FEATURES,
    FEATURE_DOMAINS,
    FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
)


def _print_section(title: str, stream: TextIO) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", file=stream)


def _print_feature_classification(stream: TextIO) -> None:
    _print_section("CLASSIFICACAO DAS FEATURES", stream)
    print(f"Numericas ({len(NUMERIC_FEATURES)}): {list(NUMERIC_FEATURES)}", file=stream)
    print(f"Binarias/categoricas ({len(BINARY_FEATURES)}): {list(BINARY_FEATURES)}", file=stream)
    print(f"Ordinais ({len(ORDINAL_FEATURES)}): {list(ORDINAL_FEATURES)}", file=stream)
    print(f"Target: {TARGET}", file=stream)


def _print_categorical_frequencies(dataframe: pd.DataFrame, stream: TextIO) -> None:
    _print_section("VALORES E FREQUENCIAS DOS DOMINIOS", stream)
    for column in FEATURE_DOMAINS:
        print(f"\n{column} (esperado: {sorted(FEATURE_DOMAINS[column])})", file=stream)
        counts = dataframe[column].value_counts(dropna=False).sort_index()
        print(counts.to_string(), file=stream)


def _print_numeric_statistics(dataframe: pd.DataFrame, stream: TextIO) -> None:
    _print_section("ESTATISTICAS DAS FEATURES NUMERICAS", stream)
    statistics = dataframe.loc[:, NUMERIC_FEATURES].describe().T
    statistics = statistics.rename(columns={"50%": "median"})
    print(
        statistics.loc[:, ["count", "mean", "std", "min", "median", "max"]].to_string(),
        file=stream,
    )

    negative_counts = (dataframe.loc[:, NUMERIC_FEATURES] < 0).sum()
    print("\nValores negativos por feature:", file=stream)
    print(negative_counts.to_string(), file=stream)


def _print_duplicate_scope(
    label: str, summary: DuplicateScopeSummary, stream: TextIO
) -> None:
    print(f"\n{label}", file=stream)
    print(f"Linhas duplicadas excedentes: {summary.duplicate_rows}", file=stream)
    print(f"Grupos de duplicatas: {summary.duplicate_groups}", file=stream)
    print(f"Tamanho de cada grupo: {list(summary.group_sizes)}", file=stream)
    print(f"Registros unicos apos remocao hipotetica: {summary.unique_rows}", file=stream)


def _print_duplicate_analysis(dataframe: pd.DataFrame, stream: TextIO) -> None:
    analysis = analyze_duplicates(dataframe)
    _print_section("ANALISE DE DUPLICATAS", stream)
    _print_duplicate_scope(
        "A) Todas as 14 colunas, incluindo Churn", analysis.all_columns, stream
    )
    _print_duplicate_scope(
        "B) Somente as 13 features, ignorando Churn", analysis.features_only, stream
    )
    print(
        f"\nGrupos com features identicas e Churn conflitante: "
        f"{analysis.conflicting_feature_groups}",
        file=stream,
    )
    print(f"Linhas pertencentes a esses grupos: {analysis.conflicting_rows}", file=stream)
    if analysis.conflict_examples.empty:
        print("Exemplos: nenhum conflito encontrado.", file=stream)
    else:
        print("Exemplos (ate 10 linhas):", file=stream)
        print(analysis.conflict_examples.to_string(index=False), file=stream)


def print_dataset_report(dataframe: pd.DataFrame, stream: TextIO = sys.stdout) -> None:
    """Print a complete report without mutating the supplied dataframe."""
    validate_dataset_columns(dataframe)

    _print_section("DIMENSOES E COLUNAS", stream)
    print(f"Linhas: {dataframe.shape[0]}", file=stream)
    print(f"Colunas: {dataframe.shape[1]}", file=stream)
    print(f"Nomes exatos: {dataframe.columns.tolist()}", file=stream)
    extra_columns = [column for column in dataframe.columns if column not in REQUIRED_COLUMNS]
    print(f"Colunas extras (nao utilizadas pelo modelo): {extra_columns}", file=stream)

    _print_section("DTYPES", stream)
    print(dataframe.dtypes.to_string(), file=stream)

    _print_section("PRIMEIRAS 5 LINHAS", stream)
    print(dataframe.head(5).to_string(index=False), file=stream)

    _print_section("ULTIMAS 5 LINHAS", stream)
    print(dataframe.tail(5).to_string(index=False), file=stream)

    _print_section("QUALIDADE DOS DADOS", stream)
    print("Valores nulos por coluna:", file=stream)
    print(dataframe.isna().sum().to_string(), file=stream)
    print(f"\nLinhas duplicadas: {int(dataframe.duplicated().sum())}", file=stream)
    print("\nValores unicos por coluna:", file=stream)
    print(dataframe.nunique(dropna=False).to_string(), file=stream)

    _print_categorical_frequencies(dataframe, stream)
    validate_dataset_domains(dataframe)
    print("\nValidacao dos dominios: OK", file=stream)

    _print_feature_classification(stream)

    _print_section("DISTRIBUICAO DO CHURN", stream)
    churn_counts = dataframe[TARGET].value_counts().sort_index()
    churn_percentages = dataframe[TARGET].value_counts(normalize=True).sort_index() * 100
    distribution = pd.DataFrame(
        {"quantidade": churn_counts, "percentual": churn_percentages}
    )
    print(distribution.to_string(float_format=lambda value: f"{value:.2f}%"), file=stream)

    _print_numeric_statistics(dataframe, stream)
    _print_duplicate_analysis(dataframe, stream)


def main() -> None:
    try:
        dataframe = read_dataset(DATASET_PATH)
        print_dataset_report(dataframe)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
