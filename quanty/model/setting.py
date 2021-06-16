import numpy as np

def base_params(db):
    
    return {
        'start': '2002-12-31', 
        'end': '2018-08-31', #'2018-03-31', 
        'db': db, 
        'assets': None, 
        'price': {'price_src':'reprice', 'trade_tol':'at_close'}, 
        'trade': {
            'trade_assets': [
                #('BND_US_Long', {'BND_US_Interm':0.3, 'BND_US_Short':0.1}), 
                #('BND_US_Long', {'BND_US_Long':1.0}), 
            ], 
            'trade_prev_nav_based': True, 
            'trade_delay': 1, 
            'freq': 'M', 
            'cash': 100000000, 
            'rebal_style': 'cum', # 'equal'
            'expense': 0.001, 
            'gr_exposure': 0.99, }, 
        'selecting': {
            'mode': 'DualMomentum', #'DualMomentum', 'AbsoluteMomentum', 'RelativeMomentum'
            'sig_mixer': {
                'sig_w_base': [1,0,0,0,0,0,1,0,0,0,0,0], #[1,0,0,0,0,0,1,0,0,0.25*4,0.25*6,0.25*12]
                'sig_w_term': 21, 
                'sig_w_dynamic': False, 
                'sig_dyn_fwd': 21*np.array([1]), 
                'sig_dyn_m_backs': 24, 
                'sig_dyn_n_sample': 63, 
                'sig_dyn_thres': 0.0, }, 
            'market': None, #'ACWI', # None도 가능
            'supporter': None, #'BND_US_Long', 
            'cash_equiv': None, #'BND_US_AGG', 
            'n_picks': None, }, 
        'weighting': {
            'w_type': 'inv_ranky2', 
            'eaa_wr': 1.0, 
            'eaa_wc': 1.0, 
            'eaa_wr_bnd': 0.5, 
            'eaa_short_period': 20, 
            'iv_period': 250, 
            'w_max': 1.0, }, 
        'stats_opts': {
            'beta_to': 'ACWI', 
            'stats_n_roll': 250, }, 
        'reinforce': {
            'follow_trend': None, #(20,60), 
            'follow_trend_market': None, #(20, 60),
            'follow_trend_supporter': None, #(20, 60), 
            'strong_condition': False, }, 
        'position_manager': {
            'losscut': 1.0, 
            'profitake_sigma': 3, 
            'rentry_sigma': 3, }, 
        'te_control': {
            'bm': None, #'ACWI', 
            'te_target': None, 
            'te_k': 0.3, 
            'te_short_period': 20, 
            'te_short_up_down_ratio_cap': True, 
            'te_short_target_cap': True, 
            'te_smoother': False, 
            'safety_buffer': 0.8, }, 
        'cash_manager': {
            'cm_method': None, #'cp', 
            'up_down_ratio_period': 20, 
            'kelly_type': 'semivariance', 
            'kelly_vol_period': 250, 
            'kelly_self_eval': True, }
    }



assets_adm = {
    'US', 
    'Smallcap_xUS', 
}


assets_global_multiasset2 = {
    #'ACWI', 
    'US_Total', 
    #'Global', 
    #'EU', 
    'Developed', 
    'EM', 
    'Asia_xJp', 
    'BRIC', 
    'Frontier', 
    'Latam', 
    'Smallcap_xUS', 
    
    #'Comdty', 
    #'WTI', 
    #'Gold', 
    #'Silver', 
    'Agriculture', 
    'NatGas', 
    'Engy', 
    'BaseMetal', 
    'PrecMetal', 

#    'BND_US_Interm',
#    'BND_US_Long', 
#    'BND_US_TIP',
    'BND_US_IG',
    'BND_US_HY',
    'BND_US_MBS',
#    'BND_US_Muni',
#    'BND_US_Bankloan',
    'BND_US_CB',
    'BND_US_HY_Muni',
    'BND_US_Pref',

#    'BND_GlobalSov_loc',
#    'BND_GlobalTip_loc',
#    'BND_GlobalSov_loch',
#    'BND_DevSov_loc',
#    'BND_DevFloat_usd',
#    'BND_DevIG_loc',
#    'BND_DevHY_loc',
#    'BND_EmSov_usd',
#    'BND_EmSov_loc',
#    'BND_EmHy_usd',
    
    #'USD_UP', 
    #'USD_DOWN', 

    #'REIT_Global', 
    'REIT_US', 
    'REIT_US_Mort',    
}



assets_global_multiasset = {
    #'ACWI', 
    'US_Total', 
    'Global', 
    'EU', 
    'Developed', 
    'EM', 
    'Asia_xJp', 
    'BRIC', 
    'Frontier', 
    'Latam', 
    'Smallcap_xUS', 
    
    #'Comdty', 
    #'WTI', 
    #'Gold', 
    #'Silver', 
    'Agriculture', 
    'NatGas', 
    'Engy', 
    'BaseMetal', 
    'PrecMetal', 

    'BND_US_Interm',
    'BND_US_Long', 
    'BND_US_TIP',
    'BND_US_IG',
    'BND_US_HY',
    'BND_US_MBS',
    'BND_US_Muni',
    'BND_US_Bankloan',
    'BND_US_CB',
    'BND_US_HY_Muni',
    'BND_US_Pref',

    'BND_GlobalSov_loc',
    'BND_GlobalTip_loc',
    'BND_GlobalSov_loch',
    'BND_DevSov_loc',
    'BND_DevFloat_usd',
    'BND_DevIG_loc',
    'BND_DevHY_loc',
    'BND_EmSov_usd',
    'BND_EmSov_loc',
    'BND_EmHy_usd',
    
    #'KTB10Y', 
    #'KTB10YL', 

    'USD_UP', 
    #'JPY', 
    #'EUR', 
    #'AUD', 
    #'CAD', 
    #'CHF', 
    #'GBP', 
    'USD_DOWN', 

    'REIT_Global', 
    'REIT_US', 
    'REIT_US_Mort',    
}



assets_region = {
    'ACWI', 
    #'US_Total', 
    #'US_lev'
    #'US_Nasdaq', 
    #'US_Nasdaq_lev', 
    #'Global', 
    #'EU', 
    'Developed', 
    'EM', 
    #'Asia_xJp', 
    #'BRIC', 
    #'Frontier', 
    #'Latam', 
    #'Smallcap_xUS', 
}

assets_fx = {
    'USD_UP', 
    'JPY', 
    'EUR', 
    'AUD', 
    'CAD', 
    'CHF', 
    'GBP', 
    'USD_DOWN', 
}


assets_alt = {
    'BND_US_TIP', 
    #'BND_US_Long', 
    
    'Comdty', 
    #'WTI', 
    #'Gold', 
    #'Silver',
    #'NatGas',    
    'Agriculture', 
    'Engy', 
    'BaseMetal', 
    'PrecMetal', 
    
    'REIT_Global', 
    'REIT_US', 
    'REIT_US_Mort',    
    'Infra', 

    'MLP', 
    #'NaturalGas', 
    #'Exploration', 
    'Solar', 
    'Wind',    
    
    'PE', 
    'Bio', 
    'Water', 

    'Lithium', 
    'Steel', 
    'Timber', 
    #'Uranium', 

    #'USD_UP', 
    #'JPY', 
    #'EUR', 
    #'AUD', 
    #'CAD', 
    #'CHF', 
    #'GBP', 
    #'USD_DOWN', 
    'IPO', 
    'Gender', 
    'M&A', 
    'Insider', 
}

assets_comdty = {
    'Comdty', 
    'WTI', 
    'Gold', 
    'Silver', 
    'Agriculture', 
    'NatGas', 
    'Engy', 
    'BaseMetal', 
    'PrecMetal', 
}


assets_kr_factor = {
    #'SEC', 
    'ESG_kr', 
    'LowVol_kr', 
    'Growth_kr', 
    'Quality_kr', 
    'Value_kr', 
    #'K200_2', 
    #'EW_kr_2', 
    #'Midcap_kr', ##
    #'DvdGrowth_kr_2', 
    'HighDvd_kr', 
    #'KSP_2', 
    #'KTB3Y', 
    #'K200L2', 
    #'LowVol_kr_2', 
    'Momentum_kr', 
    #'Quality_kr_2', 
    'Turnaround_kr', 
    #'Value_kr_2', 
    #'KTB10Y', 
    #'KTB10YL', 
    'K200', 
    #'EW_kr', ##
    'HighBeta_kr', 
    'Contrarian_kr', 
    'DvdGrowth_kr', 
    #'DvdSust_kr', ##
    #'KSP', ##
    #'HighDvd_kr_2', 
    'K200L', 
    #'K200inv', 
    #'LowVol_kr_3', 
    'MomentumGrowth_kr', 
    #'Pref_kr', ##
}

assets_kr_factor2 = {
    'ESG_kr', 
    'LowVol_kr_2', 
    'Quality_kr_2', 
    'Value_kr_2', 
    'HighDvd_kr', 
    'K200', 
    'HighBeta_kr', 
    'Contrarian_kr', 
    'DvdGrowth_kr', 
    'K200L', 
    'MomentumGrowth_kr', 
}


assets_us_factor = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
    #'Momentum_EM', #이거 자체의 성과가 너무 최악
    'DvdApprec', 
    'HighDvd', 
    'HighBeta', 
    'LowBeta', 
    'LowVol', 
    'HighFCF', 
    'Defensive', 
    'EW', 
    '130/30', 
    'Gender', 
    'CoveredCall', 
    'HedgefundHold', 
    'Moat', 
    'LongShort', 
    #'ManagedFut',  #이거 자체의 성과가 너무 별로임
    'M&A', 
    'ESG', 
    'IPO', 
    'Insider', 
    #'Insider2', 
    'SmallCap', 
    'Xrate_Lowvol',
}

assets_us_factor2 = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
    'DvdApprec', 
    'HighDvd', 
    #'HighBeta', 
    #'LowBeta', 
    'LowVol', 
    'IPO', 
    'Insider2', 
#     'HighFCF', 
    'Defensive', 
    'Gender', 
#     'Moat', 
#     'M&A', 
    'ESG', 
}

assets_us_factor3 = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
    'DvdApprec', 
    'HighDvd', 
    'HighBeta', 
    'LowBeta', 
    'LowVol', 
    'Defensive', 
    'ESG', 
}

assets_us_multiasset = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
    #'Momentum_EM', #이거 자체의 성과가 너무 최악
    'DvdApprec', 
    'HighDvd', 
    'HighBeta', 
    'LowBeta', 
    'LowVol', 
    'HighFCF', 
    'Defensive', 
    'EW', 
    '130/30', 
    'Gender', 
    'CoveredCall', 
    'HedgefundHold', 
    'Moat', 
    'LongShort', 
    #'ManagedFut',  #이거 자체의 성과가 너무 별로임
    'M&A', 
    'ESG', 
    'IPO', 
    'Insider', 
    #'Insider2', 
    'SmallCap', 
    'Xrate_Lowvol',

    #'Material', 
    #'ConsumerDiscretionary', 
    #'ConsumerStaples', 
    #'Energy', 
    #'Financial', 
    #'Healthcare', 
    #'Industrial', 
    #'REIT', 
    #'Tech',
    #'Telcom', 
    #'Utility',     

    #'BND_US_AGG', #
    #'BND_US_Short',
    'BND_US_Interm',
    'BND_US_Long', #
    'BND_US_TIP',
    'BND_US_IG',
    'BND_US_HY',
    'BND_US_MBS',
    'BND_US_Muni',
    'BND_US_Bankloan',
    'BND_US_CB',
    'BND_US_HY_Muni',
    'BND_US_Pref',

    'REIT_US', 
    'REIT_US_Mort',
    
    'USD_UP', 
    'USD_DOWN', 
}


assets_global_sector = {
    #'Material_Global', 
    #'ConsumerDiscretionary_Global', 
    #'ConsumerStaples_Global', 
    #'Energy_Global', 
    #'Financial_Global', 
    #'Healthcare_Global', 
    #'Industrial_Global', 
    #'REIT_Global', 
    #'Tech_Global',
    'Telcom_Global', 
    #'Utility_Global', 

    'Automotive', 
    'ConsumerService', 
    'Gaming', 
    'Media', 
    'OnlineRetail', 
    
    # Consumer staples
    'ConsumerGoods', 
    'FoodBeverage', 
    
    # Energy
    'MLP', 
    'NaturalGas', 
    'Exploration', 
    'OilService', 
    'Solar', 
    'Wind', 
    
    # Financial
    'Bank', 
    'Broker', 
    'BDC', 
    'CapitalMarket', 
    'CommunityBank', 
    'FinancialService', 
    'Insurance', 
    'PE', 
    'RegionalBank', 
    
    # Healthcare
    'Bio', 
    'HealthcareService', 
    'MedicalDevice', 
    'Pharma', 
    
    # Industrial
    'Aerospace', 
    'Airlines', 
    'Transportation', 
    'Water', 
    
    # Material
    'Agribiz', 
    'GoldMiner', 
    'SilverMiner', 
    'Homebuilder', 
    'Lithium', 
    'MetalMining', 
    'NaturalResource', 
    'Steel', 
    'Timber', 
    'Uranium', 
    
    # Tech
    'Cloud', 
    'Internet', 
    'Cybersecurity', 
    'Networking', 
    'Semiconductor', 
    'SNS', 
    'Software', 
    
    # Telcom
    
    # Utility
    'Infra', 
    
    # REIT
    'REIT_US', 
    'REIT_US_Mort',     
}


assets_global_gics = {
    'IT_kr', 
    'Financial_kr',
    'Construction_kr', 
    'Industrial_kr', 
    'Heavy_kr', 
    'Material_kr', 
    'Healthcare_kr', 
    'Energy_kr', 
    'ConsumerDiscretionary_kr', 
    'ConsumerStaples_kr', 

    'Material', 
    'ConsumerDiscretionary', 
    'ConsumerStaples', 
    'Energy', 
    'Financial', 
    'Healthcare', 
    'Industrial', 
    'REIT', 
    'Tech',
    'Telcom', 
    'Utility',     

    'Material_Global', 
    'ConsumerDiscretionary_Global', 
    'ConsumerStaples_Global', 
    'Energy_Global', 
    'Financial_Global', 
    'Healthcare_Global', 
    'Industrial_Global', 
    'REIT_Global', 
    'Tech_Global',
    'Telcom_Global', 
    'Utility_Global', 
}


assets_kr_sector = {
    'IT_kr', 
    'Financial_kr',
    'Construction_kr', 
    'Industrial_kr', 
    'Heavy_kr', 
    'Material_kr', 
    'Healthcare_kr', 
    'Energy_kr', 
    'ConsumerDiscretionary_kr', 
    'ConsumerStaples_kr', 
}


assets_us_sector = {
    'Material', 
    'ConsumerDiscretionary', 
    'ConsumerStaples', 
    'Energy', 
    'Financial', 
    'Healthcare', 
    'Industrial', 
    #'REIT', 
    'Tech',
    'Telcom', 
    'Utility', 
}

assets_us_factor_sector = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
    #'DvdApprec', 
    'HighDvd', 
    #'HighBeta', 
    #'LowBeta', 
    'LowVol', 
    #'Defensive', 
    #'ESG', 
    
    'Material', 
    'ConsumerDiscretionary', 
    'ConsumerStaples', 
    'Energy', 
    'Financial', 
    'Healthcare', 
    'Industrial', 
    #'REIT', 
    'Tech',
    'Telcom', 
    'Utility', 
}


assets_fi2 = {
    'BND_US_Long',
    'BND_US_TIP',
    'BND_US_IG',
    'BND_US_HY',
    'BND_US_MBS',
    'BND_US_Muni',
    'BND_US_Bankloan',
    'BND_US_CB',
    'BND_US_HY_Muni',
    'BND_US_Pref',
    'BND_DevIG_loc',
    'BND_DevHY_loc',
    #'BND_EmHy_usd'
    'BND_ChinaCredit_loc', 
    'KTB10YL', 
}


assets_fi = {
    #'BND_US_AGG', #
    #'BND_US_Short',
    'BND_US_Interm',
    #'BND_US_Long', #
    'BND_US_TIP',
    'BND_US_IG',
    'BND_US_HY',
    'BND_US_MBS',
    'BND_US_Muni',
    'BND_US_Bankloan',
    'BND_US_CB',
    'BND_US_HY_Muni',
    'BND_US_Pref',

    'BND_GlobalSov_loc',
    'BND_GlobalTip_loc',
    #'BND_GlobalSov_loch',
    'BND_DevSov_loc',
    'BND_DevFloat_usd',
    'BND_DevIG_loc',
    'BND_DevHY_loc',
    'BND_EmSov_usd',
    'BND_EmSov_loc',
    'BND_EmHy_usd',
    #'BND_ChinaCredit_loc', 
    
    #'KTB10Y', 
    #'KTB10YL', 
}


assets_fi3 = {
    'BND_US_Short',
    'BND_US_IG',
    'BND_US_HY',

    'BND_DevFloat_usd',
    'BND_DevIG_loc',
    'BND_DevHY_loc',
}


assets_spy_tlt = {
    'K200', 
    'BND_US_Long_krw', 
    'REIT_US',
}


assets_global_factor = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
    'HighDvd', 
    'LowVol',     
    'US_Nasdaq', 
    'HighBeta', 
    'LowBeta', 
    'HighFCF', 
    'Defensive', 
    'Insider', 
    'SmallCap',
    ##
    'DvdApprec', ##
    'EW',
    'Gender',
    'M&A',
    'ESG',
    'IPO',
}


assets_global_eq = {
    'US',
#    'Canada',
#    'Mexico',
#    'Peru',
#    'Brazil',
#    'Argentina',
#    'UK',
#    'Spain',
#    'Germany',
#    'Italy',
#    'Egypt',
    'India',
#    'Africa',
#    'SouthAfrica',
    'Russia',
#    'Korea',    
    'Japan',
#    'China',
#    'Singapore',
#    'Australia',
#    'Austria',
#    'Belgium',
    'France',
    'HongKong',
#    'Malaysia',
#    'Netherland',
#    'Sweden',
#    'Switzerland',
#    'Taiwan',
#    'Vietnam',
#    'Poland',
#    'NewZealand',
#    'Greece',
#    'Norway',
#    'Indonesia',
#    'Philip',
#    'Thailand',
#    'Turkey',
#    'Chile',
#    'Colombia',
#    'Saudi',
}



assets_global_eq2 = {
    #'US_lev',
    'US_Nasdaq',
    'Canada',
    'Mexico',
    'Peru',
    'Brazil',
    'Argentina',
    'UK',
    'Spain',
    'Germany',
    'Italy',
    'Egypt',
    'India',
    'Africa',
    'SouthAfrica',
    'Russia',
    'Korea',    
    'Japan',
    'China',
    'China_csi300',
    'China_largecap',
    'Singapore',
    'Australia',
    'Austria',
    'Belgium',
    'France',
    'HongKong',
    'Malaysia',
    'Netherland',
    'Sweden',
    'Switzerland',
    'Taiwan',
    'Vietnam',
    'Poland',
    'NewZealand',
    'Greece',
    'Norway',
    'Indonesia',
    #'Indonesia2',
    'Philip',
    'Thailand',
    'Turkey',
    'Chile',
    'Colombia',
    'Saudi',
    'Ireland',
    'Israel',
    #'India_earning',

    
    'Frontier',
    #'Frontier2',
    'MiddleEast',
    
    #'BND_US_Long', 
    #'BND_US_Long_3x',
    #'KTB10YL',
    #'REIT_US', 
}


assets_global_eq3 = {
    'US',
    'US_Nasdaq',
    'Canada',
    'Brazil',
    'UK',
    'Spain',
    'Germany',
    'Italy',
    'India',
    'Russia',
    'Korea',    
    'Japan',
    'China',
    'Singapore',
    'Australia',
    'France',
    'HongKong',
    'Netherland',
    'Switzerland',
    'Taiwan',
    'Vietnam',
}

assets_test_0 = {
    'ACWI', 
    'US_Total', 
    #'US_lev'
    #'US_Nasdaq', 
    #'US_Nasdaq_lev', 
    'Global', 
    #'EU', 
    #'Developed', 
    'EM', 
    #'Asia_xJp', 
    #'BRIC', 
    #'Frontier', 
    #'Latam', 
    #'Smallcap_xUS', 
}


assets_test_1 = {
    'ACWI', 
    'US_Total',
    #'US_lev'
    #'US_Nasdaq', 
    #'US_Nasdaq_lev', 
    'Global', 
    #'EU', 
    'Developed', 
    'EM', 
    'Asia_xJp', 
    'BRIC', 
    'Frontier', 
    'Latam', 
    #'Smallcap_xUS', 
}

assets_test_2 = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
}


assets_test_3 = {
    'Material_Global', 
    'ConsumerDiscretionary_Global', 
    'ConsumerStaples_Global', 
    'Energy_Global', 
    'Financial_Global', 
    'Healthcare_Global', 
    'Industrial_Global', 
    #'REIT_Global', 
    'Tech_Global',
    'Telcom_Global', 
    'Utility_Global', 
}
    

assets_test_4 = {
    'ACWI', 
    'US_Total', 
    'Smallcap_xUS', 
    'Global', 
    'EU', 
    'Developed', 
    'EM', 
    'Asia_xJp', 
    'BRIC', 
    'Frontier', #
    'Latam',
    'Frontier2',
    'MiddleEast',
}    


assets_test_5 = {
    'Global', 
    'EM', 
}


assets_test_6 = {
    'Quality', 
    'Value', 
    'Growth', 
    'Momentum', 
    'Momentum_EM', #이거 자체의 성과가 너무 최악
    'DvdApprec', 
    'HighDvd', 
    'HighBeta', 
    'LowBeta', 
    'LowVol', 
    'HighFCF', 
    'Defensive', 
}
    

mapper = {
    
    # Korea factors
    'sec':                               'SEC', 
    'arirang_esg':                       'ESG_kr', 
    'arirang_lowvol':                    'LowVol_kr', 
    'arirang_mtum':                      'Growth_kr', 
    'arirang_qual':                      'Quality_kr', 
    'arirang_value':                     'Value_kr', 
    'kodex200':                          'K200_2', 
    'kodex200_ew':                       'EW_kr_2', 
    'kodex200_midcap':                   'Midcap_kr', 
    'kodex_dvd_growth':                  'DvdGrowth_kr_2', 
    'kodex_dvd_high':                    'HighDvd_kr', 
    'kodex_ksp':                         'KSP_2', 
    'kodex_ktb3y':                       'KTB3Y', 
    'kodex_lev':                         'K200L2', 
    'kodex_lowvol':                      'LowVol_kr_2', 
    'kodex_mtum_plus':                   'Momentum_kr', 
    'kodex_qual_plus':                   'Quality_kr_2', 
    'kodex_turnaround':                  'Turnaround_kr', 
    'kodex_value_plus':                  'Value_kr_2', 
    'kosef_ktb10y':                      'KTB10Y', 
    'kosef_ktb10y_lev':                  'KTB10YL', 
    'tiger200':                          'K200', 
    'tiger200_ew':                       'EW_kr', 
    'tiger_beta_plus':                   'HighBeta_kr', 
    'tiger_contrarian':                  'Contrarian_kr', 
    'tiger_dvd_growth':                  'DvdGrowth_kr', 
    'tiger_dvd_sustainable':             'DvdSust_kr', 
    'tiger_ksp':                         'KSP', 
    'tiger_ksp_dvd_high':                'HighDvd_kr_2', 
    'tiger_lev':                         'K200L', 
    'tiger_inv':                         'K200inv',
    'tiger_lowvol':                      'LowVol_kr_3', 
    'tiger_mtum':                        'MomentumGrowth_kr', 
    'tiger_pref':                        'Pref_kr', 

    
    
    
    # Korea sectors
    'tiger200_it':                       'IT_kr', 
    'tiger200_financial':                'Financial_kr',
    'tiger200_construction':             'Construction_kr', 
    'tiger200_industrial':               'Industrial_kr', 
    'tiger200_heavy':                    'Heavy_kr', 
    'tiger200_material':                 'Material_kr', 
    'tiger200_healthcare':               'Healthcare_kr', 
    'tiger200_energy':                   'Energy_kr', 
    'tiger200_consumer_disc':            'ConsumerDiscretionary_kr', 
    'tiger200_consumer_stp':             'ConsumerStaples_kr', 
    

    # US sectors
    'xlb_spdr_material':                 'Material', 
    'xly_spdr_consumer_disc':            'ConsumerDiscretionary', 
    'xlp_spdr_consumer_stp':             'ConsumerStaples', 
    'xle_spdr_energy':                   'Energy', 
    'xlf_spdr_financial':                'Financial', 
    'xlv_spdr_healthcare':               'Healthcare', 
    'xli_spdr_industrial':               'Industrial', 
    'iyr_ishares_reit':                  'REIT', 
    'xlk_spdr_tech':                     'Tech',
    'iyz_ishares_telcom':                'Telcom', 
    'xlu_spdr_util':                     'Utility', 

    # Global sectors
    'rxi_ishares_consumer_disc_global':  'ConsumerDiscretionary_Global', 
    'kxi_ishares_consumer_stp_global':   'ConsumerStaples_Global', 
    'ixc_ishares_energy_global':         'Energy_Global', 
    'ixg_ishares_financial_global':      'Financial_Global', 
    'ixj_ishares_healthcare_global':     'Healthcare_Global', 
    'exi_ishares_industrial_global':     'Industrial_Global', 
    'mxi_ishares_material_global':       'Material_Global', 
    'ixn_ishares_tech_global':           'Tech_Global', 
    'ixp_ishares_telcom_global':         'Telcom_Global', 
    'jxi_ishares_util_global':           'Utility_Global', 
    'rwx_spdr_global_reit':              'REIT_Global',     
    
    # Consumer discretionary
    'carz_firstrust_automotive':         'Automotive', 
    'iyc_ishares_consumer_service':      'ConsumerService', 
    'bjk_vaneck_gaming':                 'Gaming', 
    'pbs_invesco_media':                 'Media', 
    'ibuy_amplify_online_retail':        'OnlineRetail', 
    
    # Consumer staples
    'iyk_ishares_consumer_goods':        'ConsumerGoods', 
    'pbj_invesco_food_beverage':         'FoodBeverage', 
    
    # Energy
    'amlp_alerian_mlp':                  'MLP', 
    'fcg_firstrust_natural_gas':         'NaturalGas', 
    'xop_spdr_exploration':              'Exploration', 
    'oih_vaneck_oil_service':            'OilService', 
    'tan_invesco_solar':                 'Solar', 
    'fan_firstrust_wind':                'Wind', 
    
    # Financial
    'kbe_spdr_bank':                     'Bank', 
    'iai_ishares_broker':                'Broker', 
    'bizd_vaneck_bdc':                   'BDC', 
    'kce_spdr_capital_market':           'CapitalMarket', 
    'qaba_firstrust_community_bank':     'CommunityBank', 
    'iyg_ishares_financial_service':     'FinancialService', 
    'kie_spdr_insurance':                'Insurance', 
    'psp_invesco_pe':                    'PE', 
    'kre_spdr_regional_bank':            'RegionalBank', 
    
    # Healthcare
    'ibb_ishares_bio':                   'Bio', 
    'ihf_ishares_healthcare_service':    'HealthcareService', 
    'ihi_ishares_medical_device':        'MedicalDevice', 
    'pjp_invesco_pharma':                'Pharma', 
    
    # Industrial
    'ita_ishares_aerospace':             'Aerospace', 
    'jets_usglobal_airlines':            'Airlines', 
    'iyt_ishares_transportation':        'Transportation', 
    'cgw_invesco_water':                 'Water', 
    
    # Material
    'moo_vaneck_agribiz':                'Agribiz', 
    'gdx_vaneck_gold_miner':             'GoldMiner', 
    'sil_globalx_silver_miner':          'SilverMiner', 
    'itb_ishares_homebuilder':           'Homebuilder', 
    'lit_globalx_lithium':               'Lithium', 
    'pick_ishares_metal_mining':         'MetalMining', 
    'gunr_flexshares_natural_resource':  'NaturalResource', 
    'slx_vaneck_steel':                  'Steel', 
    'wood_ishares_timber':               'Timber', 
    'ura_globalx_uranium':               'Uranium', 
    
    # Tech
    'skyy_firstrust_cloud':              'Cloud', 
    'fdn_firstrust_internet':            'Internet', 
    'cibr_firstrust_cybersecurity':      'Cybersecurity', 
    'pxq_invesco_networking':            'Networking', 
    'soxx_ishares_semiconductor':        'Semiconductor', 
    'socl_globalx_sns':                  'SNS', 
    'igv_ishares_software':              'Software', 
    
    # Telcom
    
    # Utility
    'igf_ishares_infra':                 'Infra', 
    
    # REIT
    'vnq_vanguard_us_reit':              'REIT_US', 
    'rem_ishares_us_mortgage_reit':      'REIT_US_Mort', 

    
    # Global conturies
    'spy_spdr_us':                       'US', 
    'sso_proshares_us_lev':              'US_lev', 
    'qqq_invesco_us_nasdaq':             'US_Nasdaq', 
    'qld_proshares_us_nasdaq_lev':       'US_Nasdaq_lev', 
    'ewc_ishares_canada':                'Canada', 
    'eww_ishares_mexico':                'Mexico', 
    'epu_ishares_peru':                  'Peru', 
    'ewz_ishares_brazil':                'Brazil', 
    'argt_globalx_argentina':            'Argentina', #
    'ewu_ishares_uk':                    'UK', 
    'ewp_ishares_spain':                 'Spain', 
    'ewg_ishares_germany':               'Germany', 
    'ewi_ishares_italy':                 'Italy', 
    'egpt_vaneck_egypt':                 'Egypt', 
    'inda_ishares_india':                'India', #
    'afk_vaneck_africa':                 'Africa', 
    'eza_ishares_south_africa':          'SouthAfrica', 
    'rsx_vaneck_russia':                 'Russia', 
    'ewy_ishares_skorea':                'Korea', 
    'ewj_ishares_jp':                    'Japan', 
    'mchi_ishares_china':                'China', 
    'ews_ishares_singapore':             'Singapore', 
    'ewa_ishares_australia':             'Australia', 
    'ewo_ishares_austria':               'Austria', 
    'ewk_ishares_belgium':               'Belgium', 
    'ewq_ishares_fr':                    'France', 
    'ewh_ishares_hk':                    'HongKong', 
    'ewm_ishares_malaysia':              'Malaysia', 
    'ewn_ishares_netherland':            'Netherland', 
    'ewd_ishares_sweden':                'Sweden', 
    'ewl_ishares_switzerland':           'Switzerland', 
    'ewt_ishares_taiwan':                'Taiwan', 
    'vnm_vaneck_vietnam':                'Vietnam', 
    'epol_ishares_poland':               'Poland', 
    'enzl_ishares_newzealand':           'NewZealand', 
    'grek_globalx_greece':               'Greece', 
    'norw_globalx_norway':               'Norway', 
    'eido_ishares_indonesia':            'Indonesia', 
    'ephe_ishares_philip':               'Philip', 
    'thd_ishares_thailand':              'Thailand', 
    'tur_ishares_turkey':                'Turkey', 
    'ech_ishares_chile':                 'Chile', 
    'gxg_globalx_colombia':              'Colombia', 
    'ksa_ishares_saudi':                 'Saudi', #
    
    'ashr_deutsche_china_csi300':        'China_csi300',
    'eirl_ishares_ireland':              'Ireland',
    'eis_ishares_israel':                'Israel',
    'epi_wisdomtree_india':              'India_earning', # 요건 inda와 거의 동일
    'fxi_ishares_china_largecap':        'China_largecap',
    'idx_vaneck_indonesia':              'Indonesia2',
    


    # US factors
    'qual_ishares_qual':                 'Quality', 
    'ive_ishares_value':                 'Value', 
    'ivw_ishares_growth':                'Growth', 
    'mtum_ishares_mtum':                 'Momentum', 
    'eemo_pshares_mtum_em':              'Momentum_EM', 
    'vig_vanguard_dvd_apprec':           'DvdApprec', 
    'dvy_ishares_high_dvd':              'HighDvd', 
    'sphb_pshares_high_beta':            'HighBeta', 
    'uslb_pshares_low_beta':             'LowBeta', 
    'splv_pshares_low_vol':              'LowVol', 
    'cowz_pacer_fcf':                    'HighFCF', 
    'def_pshares_defensive':             'Defensive', 
    'eusa_ishares_ew':                   'EW', 
    'csm_pshares_130_30':                '130/30', 
    'she_spdr_gender':                   'Gender', 
    'qyld_horizons_covered_call':        'CoveredCall', 
    'gvip_gs_hedgefund_hold':            'HedgefundHold', 
    'moat_vaneck_moat':                  'Moat', 
    'ftls_firstrust_long_short':         'LongShort', 
    'wtmf_wisdomtree_mgd_futs':          'ManagedFut', 
    'mna_iq_m&a':                        'M&A', 
    'susa_ishares_esg':                  'ESG', 
    'ipo_renaissance_ipo':               'IPO', 
    'know_direxion_insider':             'Insider', 
    'nfo_pshares_insider':               'Insider2', 
    'iwm_ishares_smallcap':              'SmallCap', 
    'xrlv_pshares_xrate_sen_lowvol':     'Xrate_Lowvol', 


    # Currencies
    'uup_pshares_usd_up':                'USD_UP', 
    'fxy_cshares_jpy':                   'JPY', 
    'fxe_cshares_eur':                   'EUR', 
    'fxa_cshares_aud':                   'AUD', 
    'fxc_cshares_cad':                   'CAD', 
    'fxf_cshares_chf':                   'CHF', 
    'fxb_cshares_gbp':                   'GBP', 
    'udn_pshares_usd_down':              'USD_DOWN', 


    # Commodities
    'dbc_pshares_comdty':                'Comdty', 
    'uso_uns_wti':                       'WTI', 
    'gld_spdr_gold':                     'Gold', 
    'slv_ishares_silver':                'Silver', 
    'dba_pshares_agriculture':           'Agriculture', 
    'ung_uns_ngas':                      'NatGas', 
    'dbe_pshares_energy':                'Engy', 
    'dbb_pshares_bmetal':                'BaseMetal', 
    'gltr_etfs_pmetal':                  'PrecMetal', 


    # US bonds
    'agg_ishares_us_bd_agg':             'BND_US_AGG', 
    'bil_spdr_us_tbil':                  'BND_US_Tbill', 
    'shy_ishares_us_bd_short':           'BND_US_Short', 
    'ief_ishares_us_bd_interm':          'BND_US_Interm', 
    'tlt_ishares_us_bd_long':            'BND_US_Long', 
    'tip_ishares_us_tip':                'BND_US_TIP', 
    'lqd_ishares_us_ig':                 'BND_US_IG', 
    'hyg_ishares_us_hy':                 'BND_US_HY', 
    'mbb_ishares_us_mbs':                'BND_US_MBS', 
    'mub_ishares_us_muni':               'BND_US_Muni', 
    'bkln_pshares_us_bankloan':          'BND_US_Bankloan', 
    'cwb_spdr_us_cb':                    'BND_US_CB', 
    'hyd_vaneck_us_hy_muni':             'BND_US_HY_Muni', 
    'pff_ishares_us_pref':               'BND_US_Pref', 
    
    'shy_ishares_us_bd_short_krw':       'BND_US_Short_krw', 
    'ief_ishares_us_bd_interm_krw':      'BND_US_Interm_krw', 
    'tlt_ishares_us_bd_long_krw':        'BND_US_Long_krw', 

    'tmf_direxion_us_bd_long_3x':        'BND_US_Long_3x',
    

    # Global bonds
    'bwx_spdr_global_sov_loc':           'BND_GlobalSov_loc', 
    'wip_spdr_global_tip_loc':           'BND_GlobalTip_loc', 
    'bndx_vangard_global_sov_loc_h':     'BND_GlobalSov_loch', 
    'igov_ishares_developed_sov_loc':    'BND_DevSov_loc', 
    'flot_ishares_developed_float_usd':  'BND_DevFloat_usd', 
    'picb_pshares_developed_ig_loc':     'BND_DevIG_loc', 
    'hyxu_ishares_developed_hy_loc':     'BND_DevHY_loc', 
    'emb_ishares_em_sov_usd':            'BND_EmSov_usd', 
    'emlc_vaneck_em_sov_loc':            'BND_EmSov_loc', 
    'emhy_ishares_em_hy_usd':            'BND_EmHy_usd', 
    'dsum_pshares_china_credit_loc':     'BND_ChinaCredit_loc', 
    '114260_kodex_kr_bd_short_usd':      'BND_KR_Short_usd', 
    '148070_kosef_kr_bd_interm_usd':     'BND_KR_Interm_usd', 
    '167860_kosef_kr_bd_interm_lev_usd': 'BND_KR_Interm_lev_usd', 


    # Regions
    'acwi_ishares_acwi':                 'ACWI', 
    'vti_vanguard_us_total':             'US_Total', 
    'gwx_spdr_intl_smallcap':            'Smallcap_xUS', 
    'veu_vanguard_global':               'Global', 
    'iev_ishares_europe':                'EU', 
    'efa_ishares_developed':             'Developed', 
    'eem_ishares_em':                    'EM', 
    'aaxj_ishares_asia_xjp':             'Asia_xJp', 
    'bkf_ishares_bric':                  'BRIC', 
    'fm_ishares_frontier':               'Frontier', #
    'ilf_ishares_latam':                 'Latam',
    
    'frn_invesco_frontier':              'Frontier2',
    #'gaf_spdr_middle_east_africa':       'MiddleEast_Africa',
    'gulf_wisdomtree_middle_east':       'MiddleEast',
    
    
    # Multiasset
    'aom_ishares_global_alloc':          'Global_alloc', 
    
    
    # US sizes
    'mgc_vangard_us_mega_cap':           'US_Megacap',
    'vo_vangard_us_mid_cap':             'US_Midcap',
    'ijh_ishares_us_small_cap':          'US_Smallcap'
}