from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import HBox

from .utils import VizOptions


@dataclass
class EventExplorer:
    event_data: pd.DataFrame
    event_name_col: str = 'event_name'
    event_time_col: str = 'event_timestamp'
    max_date: datetime = None
    min_date: datetime = None
    viz_options: VizOptions = field(default_factory=VizOptions)
    coarse_grain: str = 'd' # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.floor.html
    fine_grain: str = 'min'
    
    def __post_init__(self):
        self.internal_df = self.event_data

        self.count_col = '_count'
        self.coarse_col = '_coarse_dates'
        self.fine_col = '_fine_dates'

        self.internal_df[self.count_col] = 1
    
        if self.event_name_col not in self.internal_df.columns:
            raise ValueError(f'{self.event_name_col} must be a column in the dataframe passed as event_data')
        if self.event_name_col not in self.internal_df.columns:
            raise ValueError(f'{self.event_name_col} must be a column in the dataframe passed as event_data')

        if self.max_date is None:
            self.max_date = self.internal_df[self.event_time_col].max().date()
        if self.min_date is None:
            self.min_date = self.internal_df[self.event_time_col].min().date()  

        if not self.coarse_col in self.internal_df.columns:
            self.internal_df[self.coarse_col] = self.internal_df[self.event_time_col].dt.floor(self.coarse_grain)
        if not self.fine_col in self.internal_df.columns:
            self.internal_df[self.fine_col] = self.internal_df[self.event_time_col].dt.floor(self.fine_grain)

        self.entities = [col for col, dtype in self.internal_df.dtypes.items() if dtype == 'object']
        self.event_names = self.internal_df[self.event_name_col].drop_duplicates().to_list()

        # TODO: validate the grain/min date/max date

    def plot_event_volume(self):
        main_figure = go.FigureWidget()
        focus_figure = go.FigureWidget()
        fig1, fig2 = paired_plots(
            df=self.internal_df,
            main_figure=main_figure,
            focus_figure=focus_figure,
            main_x_col=self.coarse_col, 
            focus_x_col=self.fine_col, 
            y_col=self.count_col, 
            selection_col=self.event_name_col,
            viz_options=self.viz_options
        )
        fig1.update_layout(title=go.layout.Title(text='Event Volume Over Time'))
        return HBox([fig1, fig2])

    def explore_entity(self, entity, item=None):
        if not entity in self.entities:
            raise ValueError(f'{entity} not found. Available entities: {self.entities}')

        if item is None:
            item = self.internal_df[[entity]].drop_duplicates().sample(n=1)[entity].iloc[0]

        subset = self.internal_df[self.internal_df[entity] == item]

        main_figure = go.FigureWidget()
        main_figure.layout.hovermode = 'closest'
        coarse_grain_summary = (
            subset[[self.event_name_col, self.coarse_col, self.count_col]]
            .groupby([self.event_name_col, self.coarse_col], as_index=False)
            .count()
        )

        for i, (label, data) in enumerate(coarse_grain_summary.groupby(self.event_name_col)):
            trace = go.Scatter(
                x=data[self.coarse_col],
                y=data[self.count_col],
                mode='markers',
                name=label
            )
            n = data.shape[0]
            color = self.viz_options.color_scale[i]
            trace.marker.color = [color]*n
            trace.marker.size = [self.viz_options.default_marker_size]*n
            trace.line.color = color
            main_figure.add_trace(trace)

        main_figure.update_layout(title=go.layout.Title(text=f'{str(item).title()} Events'))

        return main_figure
    
    def _add_sessions(self, df, entity, session_length):
        session_id_col = '_session_id' # this could be more efficient for non-sessionization
        df.sort_values([entity,  self.event_time_col], inplace=True)
        df['timediff'] = (
            df
            .groupby(entity)[self.event_time_col]
            .diff()  # TODO: can I change the shift logic below to work more like this?
        )
        df['_is_session_start'] = (df['timediff'].isna() | (df['timediff'] > session_length))
        df.drop(columns=['timediff'], inplace=True)
        df['_is_session_end'] = df[['_is_session_start']].shift(-1)
        df[session_id_col] = df['_is_session_start'].cumsum()
        df['_rank'] = df.groupby(session_id_col)[self.event_time_col].rank(method='first')
        return df

    #@profile
    def plot_sankey(
        self,
        entity='event_user_text_historical_escaped',
        first_event=None,
        last_event=None,
        n_events=5,
        session_length=timedelta(hours=2),
        include_duplicates=True,
        ignore_list=[],
    ):  # TODO: add session start & end (+ on specific events) options
        # TODO: sessionization should be optional
        
        df = self.internal_df.loc[:,:] 

        # check params
        if first_event is not None and last_event is not None:
            raise ValueError('pass in either first or last event, not both')
        
        if first_event is None and last_event is None:
            raise ValueError('either first or last event is required')

        if first_event is not None:
            key_event = first_event
            shift_direction = 1
        
        if last_event is not None:
            raise(NotImplementedError)
            key_event = last_event
            shift_direction = -1


        # initial filtering
        if not include_duplicates:
            raise NotImplementedError
        
        if len(ignore_list) > 0:
            df = df.loc[df[self.event_name_col].isin(set(self.event_names) - set(ignore_list)), :] # remove ignored event types


        # build sessions
        df = self._add_sessions(df, entity, session_length)

        # select events to include
        session_id_col = '_session_id' # TODO: consolidate this var
        key_events = df.loc[df[self.event_name_col] == key_event, [session_id_col, '_rank']].groupby(session_id_col, as_index=False).min()
        key_events['_key_event'] = 1
        rnk_increment = pd.DataFrame({'rank_increment': range(n_events)})
        selected_events = key_events.merge(rnk_increment, how='cross')
        selected_events['_rank'] = selected_events['_rank'] + selected_events['rank_increment']
        selected_events.drop(columns=['_key_event'], inplace=True)
        events_to_plot = df.merge(selected_events, on=[session_id_col, '_rank'], how='inner', suffixes=None)
        

        # format events for plotting

        final_df = events_to_plot[[session_id_col, self.event_name_col, 'rank_increment']]

        paired_events = pd.concat([
            final_df, 
            final_df
            .shift(-1 * shift_direction)
            .rename(columns={session_id_col: 'next_session_id', self.event_name_col: f'next_{self.event_name_col}', 'rank_increment': 'next_rank_increment'})
            ], axis=1
        )
        to_plot = (
            paired_events[paired_events[session_id_col] == paired_events[session_id_col]]
            [['rank_increment', self.event_name_col, f'next_{self.event_name_col}', 'next_rank_increment']]
            .groupby(['rank_increment', self.event_name_col, f'next_{self.event_name_col}'], as_index=False)
            .count()
            .rename(columns={'next_rank_increment': 'count'})
        )

        to_plot['source'] = to_plot['rank_increment'].astype(str) + '_' + to_plot[self.event_name_col]
        to_plot['target'] = (to_plot['rank_increment'] + 1).astype(str) + '_' + to_plot[f'next_{self.event_name_col}']

        # TODO: what? 
        to_plot = to_plot[to_plot['rank_increment'] < (n_events-1)]

        nodes = set(to_plot['source']).union(to_plot['target'])
        node_ind_dict = {k:v for k, v in zip(nodes, range(len(nodes)))}
        to_plot_indices = to_plot[['source', 'target', 'count']].replace(node_ind_dict)


        # plot

        flow = go.Sankey(
            node=dict(
                pad = 15,
                thickness = 20,
                line = dict(color = 'blue', width = 0.5),
                label = list(node_ind_dict.keys()),
                color = 'purple'
            ),
            link=dict(
                source = to_plot_indices['source'].to_list(),  
                target = to_plot_indices['target'].to_list(),
                value = to_plot_indices['count'].to_list()
        ))

        fig = go.Figure()
        fig.add_trace(flow)
        fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)

        return fig, to_plot


        

def paired_plots(
        df,
        main_figure,
        focus_figure,
        main_x_col, 
        focus_x_col, 
        y_col, 
        selection_col,
        viz_options
    ):

    def _update_focus_area(trace, points, selector, trace_ind, trace_label):
        "First 3 args required for plotly onclick function, last two will be replaced with partial"
        if len(points.point_inds) > 0:
            clicked_point = points.point_inds[-1]
            subplot_keys = {'trace': trace_ind, 'label': trace_label, 'ind': clicked_point}

            selected_date = main_figure.data[subplot_keys['trace']].x[subplot_keys['ind']]
            selected_event = subplot_keys['label']
            fine_grain_summary = (
                df[(df[main_x_col] == selected_date)
                & (df[selection_col] == selected_event)]
                [[focus_x_col, y_col]]
                .groupby(focus_x_col, as_index=False).count()
            )
            focus_figure.data[0].x = fine_grain_summary[focus_x_col]
            focus_figure.data[0].y = fine_grain_summary[y_col]
            focus_figure.update_layout(title=go.layout.Title(text=f'{selected_event.title()} Volume At {selected_date}'))

    main_figure.layout.hovermode = 'closest'

    coarse_grain_summary = (
        df[[selection_col, main_x_col, y_col]]
        .groupby([selection_col, main_x_col], as_index=False)
        .count()
    )

    for i, (label, data) in enumerate(coarse_grain_summary.groupby(selection_col)):
        trace = go.Scatter(
            x=data[main_x_col],
            y=data[y_col],
            mode='lines+markers',
            name=label
        )
        n = data.shape[0]
        color = viz_options.color_scale[i]
        trace.marker.color = [color]*n
        trace.marker.size = [viz_options.default_marker_size]*n
        trace.line.color = color
        main_figure.add_trace(trace)
        click_func = partial(_update_focus_area, trace_ind=i, trace_label=label)
        main_figure.data[i].on_click(click_func)

    focus_figure.add_trace(go.Scatter(mode='markers'))

    return [main_figure, focus_figure]
