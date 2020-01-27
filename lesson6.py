import quantopian.algorithm as algo

# Pipeline imports
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.psychsignal import stocktwits

from quantopian.pipeline.experimental import risk_loading_pipeline

from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
import quantopian.optimize as opt

def initialize(context):
    context.day_count = 0
    context.daily_message = "Day {}"
    context.weekly_message = "Time to place some trades" 
    
    context.max_leverage = 1.0
    context.max_position_size = 0.015
    context.max_turnover = 0.95
    
    
    algo.attach_pipeline(
        make_pipeline(), 
        'data_pipe'
    )
    
    algo.attach_pipeline(
        risk_loading_pipeline(), 
        'risk_pipe'
    )
    
    algo.schedule_function(
        rebalance, 
        date_rule = algo.date_rules.week_start(),
        time_rule = algo.time_rules.market_open()
    )
    
    
def make_pipeline():
    base_universe = QTradableStocksUS()
    sentiment_score = SimpleMovingAverage(
        inputs = [stocktwits.bull_minus_bear], 
        window_length = 3
    )
    
    return Pipeline(
        columns = {
            'sentiment_score': sentiment_score, 
        },
        screen=(
            base_universe & sentiment_score.notnull()
        )
    )
    
    
def before_trading_start(context, data): 
    context.pipeline_data = algo.pipeline_output('data_pipe')
    context.risk_factor_betas = algo.pipeline_output('risk_pipe')
    
    
def rebalance(context, data): 
    alpha = context.pipeline_data.sentiment_score
   
    if not alpha.empty():
        objective = opt.MaximizeAlpha(alpha) 
            #Constrain position size
        constrain_position_size = opt.PositionConcentration.with_equal_bounds(
            -context.max_position_size, 
            context.max_position_size
            )
    
        #Constrain portfolio exposure
        max_leverage = opt.MaxGrossExposure(context.max_position_size)
    
        #Ensure long and short books are roughly the same size
        dollar_neutral = opt.DollarNeutral()
        
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas, 
            version=opt.Newest
        )
        
        
        
        algo.order_optimal_portfolio(
            objective = objective, 
            constraints = [
                constrain_position_size, 
                max_leverage, 
                dollar_neutral, 
                max_turnover, 
                factor_risk_constraints,
            ])
        
        
    log.debug(context.pipeline_data.head())