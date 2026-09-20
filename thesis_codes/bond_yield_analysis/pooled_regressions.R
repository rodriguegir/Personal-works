# ============================================================
# Packages
# ============================================================

library(eurostat)
library(dplyr)
library(tidyr)
library(lubridate)
library(purrr)
library(fredr)
library(fixest)
library(ggplot2)
library(tseries)
library(urca)
library(lmtest)
library(sandwich)
library(zoo)
library(readr)


# ============================================================
# FRED API key
# ============================================================

# Set your FRED API key in your local environment:
# Sys.setenv(FRED_API_KEY = "YOUR_API_KEY")

fredr_set_key(Sys.getenv("FRED_API_KEY"))


# ============================================================
# European countries and FRED series
# ============================================================

countries <- c(
  "FR", "IT", "DE", "AT", "ES",
  "BE", "NL", "PT", "FI"
)

# FRED codes for 10-year government bond yields

fred_codes <- list(
  FR = "IRLTLT01FRM156N",
  IT = "IRLTLT01ITM156N",
  DE = "IRLTLT01DEM156N",
  AT = "IRLTLT01ATM156N",
  ES = "IRLTLT01ESM156N",
  BE = "IRLTLT01BEM156N",
  NL = "IRLTLT01NLM156N",
  PT = "IRLTLT01PTM156N",
  FI = "IRLTLT01FIM156N"
)


# ============================================================
# Import and clean 10-year government bond yields
# ============================================================

get_yield_10y <- function(country) {

  fredr(
    series_id = fred_codes[[country]],
    frequency = "q"
  ) %>%
    select(date, value) %>%
    mutate(
      geo = country,
      yield_10y = value
    ) %>%
    select(
      geo,
      time = date,
      yield_10y
    )
}

df_yields_10y <- map_df(
  countries,
  get_yield_10y
)


# ============================================================
# 1. Net financial worth
# ============================================================

df_nfw <- get_eurostat(
  "nasq_10_f_bs",
  filters = list(
    geo = countries,
    na_item = "BF90",
    unit = "MIO_EUR",
    sector = c("S14_S15", "S11"),
    finpos = "LIAB"
  ),
  time_format = "date"
)


# Households

df_nfw_hh <- df_nfw %>%
  filter(sector == "S14_S15") %>%
  rename(
    nfw_hh = values
  )


# Corporates

df_nfw_corp <- df_nfw %>%
  filter(sector == "S11") %>%
  rename(
    nfw_corp = values
  )


# ============================================================
# 2. Nominal GDP
# ============================================================

df_gdp_nom <- get_eurostat(
  "namq_10_gdp",
  filters = list(
    geo = countries,
    na_item = "B1GQ",
    unit = "CP_MEUR",
    s_adj = "NSA"
  ),
  time_format = "date"
) %>%
  select(
    geo,
    time,
    gdp_nom = values
  )


# ============================================================
# 3. Real GDP growth
# ============================================================

df_gdp_real <- get_eurostat(
  "namq_10_gdp",
  filters = list(
    geo = countries,
    na_item = "B1GQ",
    unit = "CLV10_MEUR",
    s_adj = "NSA"
  ),
  time_format = "date"
) %>%
  group_by(geo) %>%
  arrange(time) %>%
  mutate(
    gdp_real_growth =
      (values / lag(values, 4) - 1) * 100
  ) %>%
  select(
    geo,
    time,
    gdp_real_growth
  )


# ============================================================
# 4. Inflation
# ============================================================

# HICP year-on-year inflation,
# aggregated from monthly to quarterly

df_hicp <- get_eurostat(
  "prc_hicp_manr",
  filters = list(
    geo = countries,
    coicop = "CP00"
  ),
  time_format = "date"
)

df_infl_q <- df_hicp %>%
  mutate(
    quarter = floor_date(time, "quarter")
  ) %>%
  group_by(
    geo,
    quarter
  ) %>%
  summarise(
    inflation = mean(values, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(
    time = quarter
  )


# ============================================================
# 5. Government debt
# ============================================================

df_debt <- get_eurostat(
  "gov_10q_ggdebt",
  filters = list(
    geo = countries,
    unit = "MIO_EUR",
    sector = "S13",
    na_item = "GD"
  ),
  time_format = "date"
) %>%
  rename(
    debt = values
  )


# ============================================================
# 6. Short-term interest rate
# ============================================================

# Euro area 3-month money market rate

df_rate_ea <- get_eurostat(
  "irt_st_q",
  time_format = "date"
) %>%
  filter(
    geo == "EA",
    int_rt == "IRT_M3"
  ) %>%
  rename(
    rate_short = values,
    time = TIME_PERIOD
  ) %>%
  select(
    geo,
    time,
    rate_short
  )


# Apply the euro area rate to each country

df_rate_q <- map_df(
  countries,
  ~ df_rate_ea %>%
    mutate(geo = .x)
)


# ============================================================
# 7. Merge European data
# ============================================================

df_nfw_hh <- df_nfw_hh %>%
  select(
    geo,
    time,
    nfw_hh
  )

df_nfw_corp <- df_nfw_corp %>%
  select(
    geo,
    time,
    nfw_corp
  )

df_gdp_nom <- df_gdp_nom %>%
  select(
    geo,
    time,
    gdp_nom
  )

df_gdp_real <- df_gdp_real %>%
  select(
    geo,
    time,
    gdp_real_growth
  )

df_infl_q <- df_infl_q %>%
  select(
    geo,
    time,
    inflation
  )

df_debt <- df_debt %>%
  select(
    geo,
    time,
    debt
  )

df_rate_q <- df_rate_q %>%
  select(
    geo,
    time,
    rate_short
  )


# Merge all European series

df_final <- df_nfw_hh %>%

  left_join(
    df_nfw_corp,
    by = c("geo", "time")
  ) %>%

  left_join(
    df_gdp_nom,
    by = c("geo", "time")
  ) %>%

  left_join(
    df_gdp_real,
    by = c("geo", "time")
  ) %>%

  left_join(
    df_infl_q,
    by = c("geo", "time")
  ) %>%

  left_join(
    df_debt,
    by = c("geo", "time")
  ) %>%

  left_join(
    df_rate_q,
    by = c("geo", "time")
  ) %>%

  left_join(
    df_yields_10y,
    by = c("geo", "time")
  ) %>%

  mutate(

    gdp_rolling_annual =
      zoo::rollsum(
        gdp_nom,
        4,
        fill = NA,
        align = "right"
      ),

    debt_gdp =
      debt / gdp_rolling_annual * 100,

    nfw_private =
      nfw_hh + nfw_corp,

    nfw_private_gdp =
      nfw_private / gdp_rolling_annual * 100,

    nfw_private_debt =
      (nfw_hh + nfw_corp) / debt,

    ryield_10y =
      yield_10y - inflation
  ) %>%

  arrange(
    geo,
    time
  )


# ============================================================
# Growth rates and changes
# ============================================================

df_final <- df_final %>%

  group_by(geo) %>%

  mutate(

    ln_nfw_private_gdp_g =
      log(nfw_private_gdp) -
      log(lag(nfw_private_gdp, 1)),

    ln_debt_gdp_g =
      log(debt_gdp) -
      log(lag(debt_gdp, 1)),

    ln_ryield_10y_g =
      log(ryield_10y) -
      log(lag(ryield_10y, 1)),

    ln_yield_10y_g =
      log(yield_10y) -
      log(lag(yield_10y, 1)),

    ln_rate_short_g =
      log(rate_short) -
      log(lag(rate_short, 1)),

    ln_inflation_g =
      log(inflation) -
      log(lag(inflation, 1))

  ) %>%

  ungroup()


# ============================================================
# 8. US data
# ============================================================

series_list <- list(

  cd_3m =
    fredr(
      series_id = "IR3TCD01USM156N",
      frequency = "q"
    ),

  hh_net_worth =
    fredr(
      series_id = "TNWBSHNO",
      frequency = "q"
    ),

  debt_gdp =
    fredr(
      series_id = "GFDEGDQ188S",
      frequency = "q"
    ),

  corp_net_worth =
    fredr(
      series_id = "TNWMVBSNNCB",
      frequency = "q"
    ),

  core_sticky_cpi =
    fredr(
      series_id = "CORESTICKM159SFRBATL",
      frequency = "q"
    ),

  real_gdp =
    fredr(
      series_id = "GDPC1",
      frequency = "q"
    ),

  foreign_debt_share =
    fredr(
      series_id = "HBFIGDQ188S",
      frequency = "q"
    ),

  yield_10y =
    fredr(
      series_id = "DGS10",
      frequency = "q"
    ),

  yield_3m =
    fredr(
      series_id = "DTB3",
      frequency = "q"
    ),

  nominal_gdp =
    fredr(
      series_id = "GDP",
      frequency = "q"
    ),

  houst =
    fredr(
      series_id = "HOUST",
      frequency = "q"
    ),

  mkt_value_debt =
    fredr(
      "MVMTD027MNFRBDAL",
      frequency = "q"
    ),

  spread_10y_aaa =
    fredr(
      "AAA10Y",
      frequency = "q"
    ),

  spread_10y_baa =
    fredr(
      "BAA10Y",
      frequency = "q"
    ),

  yield_aaa =
    fredr(
      "AAA",
      frequency = "q"
    ),

  yield_long_t =
    fredr(
      "LTGOVTBD",
      frequency = "q"
    ),

  ca_balance_gdp =
    fredr(
      "USAB6BLTT02STSAQ",
      frequency = "q"
    ),

  surplus =
    fredr(
      "M318501Q027NBEA",
      frequency = "q"
    ),

  emp_ratio_25_54 =
    fredr(
      "LNS12300060",
      frequency = "q"
    ),

  em_rat =
    fredr(
      "EMRATIO",
      frequency = "q"
    ),

  hp_infla =
    fredr(
      "USSTHPI",
      frequency = "q"
    ),

  div =
    fredr(
      "TNWBSHNO",
      frequency = "q"
    ),

  dollar =
    fredr(
      "DTWEXBGS",
      frequency = "q"
    ),

  dollarjpy =
    fredr(
      "DEXJPUS",
      frequency = "q"
    )
)


# ============================================================
# Clean US series
# ============================================================

clean_list <- imap(
  series_list,
  ~ .x %>%
    select(
      date,
      value
    ) %>%
    rename(
      !!.y := value
    )
)


# ============================================================
# Construct US macro dataset
# ============================================================

macro_df <- reduce(
  clean_list,
  full_join,
  by = "date"
) %>%

  arrange(date) %>%

  mutate(

    log_cd_3m_g =
      log(
        cd_3m /
          lag(cd_3m, 1)
      ),

    real_gdp_growth =
      (
        real_gdp -
          lag(real_gdp)
      ) /
      lag(real_gdp) * 100,

    net_wealth_gdp =
      (
        hh_net_worth +
          corp_net_worth
      ) /
      nominal_gdp,

    log_net_wealth_gdp_g =
      log(
        net_wealth_gdp /
          lag(net_wealth_gdp, 1)
      ),

    log_emp_ratio_25_54_g =
      log(
        emp_ratio_25_54 /
          lag(emp_ratio_25_54, 1)
      ),

    log_debt_gdp_g =
      log(
        debt_gdp /
          lag(debt_gdp, 1)
      ),

    mkt_value_debt_gdp =
      mkt_value_debt /
      nominal_gdp,

    wealth_debt =
      net_wealth_gdp /
      debt_gdp,

    log_wealth_debt =
      log(wealth_debt),

    wealth_mkt_debt =
      (
        hh_net_worth +
          corp_net_worth
      ) /
      mkt_value_debt,

    log_wealth_debt_g =
      log(
        wealth_debt /
          lag(wealth_debt)
      ),

    log_wealth_mkt_debt =
      log(wealth_mkt_debt),

    log_wealth_mkt_debt_g =
      log(
        wealth_mkt_debt /
          lag(wealth_mkt_debt, 1)
      ),

    d_wealth_mkt_debt =
      wealth_mkt_debt -
      lag(wealth_mkt_debt),

    spread_manu_aaa =
      yield_aaa -
      yield_long_t,

    d_spread_manu_aaa =
      spread_manu_aaa -
      lag(spread_manu_aaa),

    log_spread_manu_aaa =
      log(spread_manu_aaa),

    log_spread_manu_aaa_g =
      log(
        spread_manu_aaa /
          lag(spread_manu_aaa, 1)
      ),

    vol =
      yield_10y -
      yield_3m,

    surplus_gdp =
      surplus /
      nominal_gdp,

    s_min_inv_priv_gdp =
      ca_balance_gdp -
      surplus_gdp,

    s_min_priv_debt =
      s_min_inv_priv_gdp /
      debt_gdp,

    slope_yield =
      yield_10y -
      yield_3m,

    log_spread_10y_aaa_g =
      log(
        spread_10y_aaa /
          lag(spread_10y_aaa, 1)
      ),

    log_yield_10y_g =
      log(
        yield_10y /
          lag(yield_10y, 1)
      ),

    log_yield_long_t_g =
      log(
        yield_long_t /
          lag(yield_long_t, 1)
      ),

    log_houst_g =
      log(
        houst /
          lag(houst, 1)
      ),

    log_em_rat_g =
      log(
        em_rat /
          lag(em_rat, 1)
      ),

    log_hp_infla_g =
      log(
        hp_infla /
          lag(hp_infla, 1)
      ),

    log_div_g =
      log(
        div /
          lag(div, 1)
      ),

    log_foreign_debt_share_g =
      log(
        foreign_debt_share /
          lag(foreign_debt_share, 1)
      ),

    log_dollar_g =
      log(
        dollar /
          lag(dollar, 1)
      ),

    log_dollarjpy_g =
      log(
        dollarjpy /
          lag(dollarjpy, 1)
      ),

    log_inflation_g =
      log(
        core_sticky_cpi /
          lag(core_sticky_cpi, 1)
      )
  )


# ============================================================
# 9. Construct US panel observations
# ============================================================

df_us <- macro_df %>%

  transmute(

    geo = "US",

    time =
      floor_date(
        date,
        "quarter"
      ),

    ln_yield_10y_g =
      log_yield_10y_g,

    ln_rate_short_g =
      log_cd_3m_g,

    ln_nfw_private_gdp_g =
      log_net_wealth_gdp_g,

    gdp_real_growth =
      real_gdp_growth,

    inflation =
      core_sticky_cpi,

    ln_inflation_g =
      log_inflation_g,

    ln_debt_gdp_g =
      log_debt_gdp_g
  ) %>%

  arrange(time)


# ============================================================
# 10. European-US panel
# ============================================================

df_panel <- bind_rows(

  df_final %>%
    select(
      geo,
      time,
      ln_yield_10y_g,
      ln_rate_short_g,
      ln_nfw_private_gdp_g,
      gdp_real_growth,
      ln_inflation_g,
      inflation,
      ln_debt_gdp_g
    ),

  df_us
)


# ============================================================
# 11. Country subsamples
# ============================================================

df_fr_us_de_it_es <- df_panel %>%
  filter(
    geo %in%
      c(
        "US",
        "DE",
        "FR",
        "ES",
        "IT"
      )
  )


df_fr_de_it_es <- df_final %>%
  filter(
    geo %in%
      c(
        "DE",
        "FR",
        "ES",
        "IT"
      )
  )


df_fr_de_it_es_nl <- df_panel %>%
  filter(
    geo %in%
      c(
        "DE",
        "FR",
        "ES",
        "IT",
        "NL"
      )
  )


df_fr_de_es_nl <- df_panel %>%
  filter(
    geo %in%
      c(
        "DE",
        "FR",
        "ES",
        "NL"
      )
  )


df_fr_de_nl <- df_panel %>%
  filter(
    geo %in%
      c(
        "DE",
        "FR",
        "NL"
      )
  )


df_fr_de <- df_panel %>%
  filter(
    geo %in%
      c(
        "DE",
        "FR"
      )
  )


# ============================================================
# 12. Panel regressions
# ============================================================


# ------------------------------------------------------------
# Full panel - country fixed effects
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_panel,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Full panel - country and time fixed effects
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    ln_inflation_g |

    geo + time,

  data = df_panel,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# US, Germany, France, Spain and Italy
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_fr_us_de_it_es,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# US, Germany, France, Spain and Italy
# Country + time fixed effects
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo + time,

  data = df_fr_us_de_it_es,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France, Spain and Italy
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_fr_de_it_es,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France, Spain and Italy
# Country + time fixed effects
#
# Short-term rate excluded because it is
# collinear with time fixed effects
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    gdp_real_growth +
    ln_inflation_g |

    geo + time,

  data = df_fr_de_it_es,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France, Spain, Italy and Netherlands
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_fr_de_it_es_nl,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France, Spain, Italy and Netherlands
# Country + time fixed effects
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    gdp_real_growth +
    ln_inflation_g |

    geo + time,

  data = df_fr_de_it_es_nl,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France, Spain and Netherlands
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_fr_de_es_nl,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France, Spain and Netherlands
# Country + time fixed effects
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    gdp_real_growth +
    ln_inflation_g |

    geo + time,

  data = df_fr_de_es_nl,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France and Netherlands
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_fr_de_nl,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France and Netherlands
# Country + time fixed effects
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    gdp_real_growth +
    ln_inflation_g |

    geo + time,

  data = df_fr_de_nl,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany and France
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_fr_de,

  vcov = "hetero"
)

summary(model)


# ------------------------------------------------------------
# Germany, France, Italy and Spain
# ------------------------------------------------------------

model <- feols(

  ln_yield_10y_g ~
    ln_debt_gdp_g +
    ln_nfw_private_gdp_g +
    ln_rate_short_g +
    gdp_real_growth +
    ln_inflation_g |

    geo,

  data = df_fr_de_it_es,

  vcov = "hetero"
)

summary(model)
