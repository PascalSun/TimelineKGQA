import loguru
import numpy as np
import plotly.graph_objects as go

logger = loguru.logger


class AllenTemporalSimulator:
    def __init__(self, n, m, max_time=10, mode="timeranges"):
        self.n = n  # number of random points to generate
        self.m = m  # number of points to generate around each random point
        self.max_time = max_time  # maximum time value for all axes
        self.mode = mode  # mode to generate points, either "timeranges" or "timepoint"
        self.color_shape_map = {
            "before": ("#FF0000", "circle"),  # bright red, circle
            "after": ("#FF4500", "square"),  # orange-red, square
            "meets": ("#1E90FF", "circle-open"),  # dodger blue, circle-open
            "met_by": ("#4169E1", "square-open"),  # royal blue, square-open
            "overlaps": ("#32CD32", "cross"),  # lime green, cross
            "overlapped_by": ("#228B22", "x"),  # forest green, x
            "starts": ("#800080", "diamond"),  # purple, diamond
            "started_by": ("#9932CC", "diamond-open"),  # dark orchid, diamond-open
            "during": ("#FFA500", "circle"),  # orange, circle
            "contains": ("#FF8C00", "square"),  # dark orange, square
            "finishes": ("#FFFF00", "circle-open"),  # yellow, circle-open
            "finished_by": ("#FFD700", "square-open"),  # gold, square-open
            "equals": ("#00FFFF", "diamond"),  # cyan, diamond
        }

    def generate_surface(self):
        """
        This is the 3D surface that represents all information temporal information
        """
        x = np.linspace(0, self.max_time, 100)
        y = np.linspace(0, self.max_time, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.maximum(Y - X, 0)  # Z starts from 0 at the origin and increases
        # remove all points where X > Y
        Z[X > Y] = np.nan
        return X, Y, Z

    def generate_nearby_time_ranges(self, point):
        start, end = point
        nearby_points = []
        for _ in range(self.m):
            # random select a relationship to generate a nearby point
            temporal_relation = np.random.choice(list(self.color_shape_map.keys()))
            if temporal_relation == "meets":
                new_start = end
                new_end = np.random.uniform(new_start + 0.1, self.max_time)
            elif temporal_relation == "met_by":
                new_end = start
                new_start = np.random.uniform(0, new_end - 0.1)
            elif temporal_relation == "starts":
                new_start = start
                new_end = np.random.uniform(start + 0.1, self.max_time)
            elif temporal_relation == "started_by":
                new_end = end
                new_start = np.random.uniform(0, new_end - 0.1)
            elif temporal_relation == "finishes":
                new_end = end
                new_start = np.random.uniform(0, new_end - 0.1)
            elif temporal_relation == "finished_by":
                new_start = start
                new_end = np.random.uniform(start + 0.1, self.max_time)
            elif temporal_relation == "equals":
                new_start = start
                new_end = end
            else:
                new_start = np.random.uniform(
                    max(0, start - 100), min(self.max_time, start + 100)
                )
                new_end = np.random.uniform(
                    max(new_start + 0.1, end - 100), min(self.max_time, end + 100)
                )
            nearby_points.append((new_start, new_end))
        return nearby_points

    def generate_nearby_timepoints(self, point):
        start, end = point
        nearby_points = []
        for _ in range(self.m):
            # random select a relationship to generate a nearby point
            temporal_relation = np.random.choice(["before", "equals", "after"])
            if temporal_relation == "before":
                new_start = np.random.uniform(0, start)
                new_end = new_start
            elif temporal_relation == "after":
                new_start = np.random.uniform(end, self.max_time)
                new_end = new_start
            elif temporal_relation == "equals":
                new_start = start
                new_end = end
            nearby_points.append((new_start, new_end))
        return nearby_points

    def generate_nearby_timepoints_for_timerange(self, time_range):
        start, end = time_range
        nearby_points = []
        for _ in range(self.m):
            # random select a relationship to generate a nearby point
            temporal_relation = np.random.choice(
                ["before", "finishes", "contains", "started_by", "after"]
            )
            if temporal_relation == "before":
                new_start = np.random.uniform(0, start)
                new_end = new_start
            elif temporal_relation == "finishes":
                new_start = start
                new_end = new_start
            elif temporal_relation == "contains":
                new_start = np.random.uniform(start, end)
                new_end = new_start
            elif temporal_relation == "started_by":
                new_start = end
                new_end = new_start
            elif temporal_relation == "after":
                new_start = np.random.uniform(end, self.max_time)
                new_end = new_start
            else:
                return
            nearby_points.append((new_start, new_end))
        return nearby_points

    def determine_relation(self, interval1, interval2):
        """
        Determine the relationship between two temporal intervals

        """
        start1, end1 = interval1
        start2, end2 = interval2

        if end1 < start2:
            return "before"  # interval1 is before interval2
        elif start1 > end2:
            return "after"  # interval1 is after interval2
        elif end1 == start2:  # interval1 meets interval2
            return "meets"
        elif start1 == end2:  # interval1 is met by interval2
            return "met_by"
        elif start1 < start2 < end1 < end2:  # interval1 overlaps interval2
            return "overlaps"
        elif start2 < start1 < end2 < end1:  # interval1 is overlapped by interval2
            return "overlapped_by"
        elif start1 == start2 and end1 < end2:  # interval1 starts interval2
            return "starts"
        elif start1 == start2 and end1 > end2:  # interval1 is started by interval2
            return "started_by"
        elif start2 < start1 < end1 < end2:  # interval1 is during interval2
            return "during"
        elif start1 < start2 < end2 < end1:  # interval1 contains interval2
            return "contains"
        elif start1 < start2 < end2 == end1:  # interval1 finishes interval2
            return "finishes"
        elif start2 < start1 < end1 == end2:  # interval1 is finished by interval2
            return "finished_by"
        elif start1 == start2 and end1 == end2:  # interval1 equals interval2
            return "equals"

    def determine_relation_timepoint(self, point1, point2):
        start1, end1 = point1
        start2, end2 = point2
        if end1 < start2:
            return "before"
        elif start1 > end2:
            return "after"
        elif start1 == start2 and end1 == end2:
            return "equals"

    def determine_relation_timepointrange(self, point1, point2):
        """
        Point 1 will be a time point, and point 2 will be a time range
        So there will be 5 relationships: before, starts, during, finishes, after

        """
        start1, end1 = point1
        start2, end2 = point2

        if end1 < start2:
            return "before"
        elif start1 == start2 and end1 < end2:
            return "starts"
        elif start1 > start2 and end1 < end2:
            return "during"
        elif start1 < start2 and end1 == end2:
            return "finishes"
        elif start1 > start2:
            return "after"

    def determine_relation_timerangepoint(self, point1, point2):
        """
        Point 1 will be a time range, and point 2 will be a time point
        So there will be 5 relationships: before, meets, contain, finished_by, after
        """

        start1, end1 = point1
        start2, end2 = point2  # point2 is a time point, so end2 = start2

        if end1 < start2:
            return "before"  # range1 is before point2
        elif end1 == start2:
            return "finished_by"  # range 1 meets point2
        elif start1 < start2 and end1 > start2:
            return "contains"  # range 1 contains point2
        elif start1 == start2:
            return "met_by"  # range 1 finished_by point2
        elif start1 > start2:
            return "after"  # range 1 is after point2

    def visualize(self):
        """
        First generate the 3D surface for the visualization
        """
        X, Y, Z = self.generate_surface()

        fig = go.Figure()

        if self.mode == "surface":
            # show the surface track
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    # give it the grey color
                    colorscale=[[0, "rgb(220,220,220)"], [1, "rgb(220,220,220)"]],
                    showscale=False,
                    opacity=0.5,
                    showlegend=False,
                )
            )

        traces = {
            relation: go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=5, color=color, symbol=shape),
                name=relation,
            )
            for relation, (color, shape) in self.color_shape_map.items()
        }

        """
        Here is for relationship between temporal intervals
        """
        if self.mode == "timeranges":
            # define a center point
            center_point = (25, 75)
            start, end = center_point
            fig.add_trace(
                go.Scatter3d(
                    x=[start],
                    y=[end],
                    z=[end - start],
                    mode="markers",
                    marker=dict(size=20, color="black", symbol="cross"),
                    showlegend=False,
                )
            )

            nearby_points = self.generate_nearby_time_ranges(center_point)
            for nearby_point in nearby_points:
                n_start, n_end = nearby_point
                relation = self.determine_relation(center_point, nearby_point)
                traces[relation].x = traces[relation].x + (n_start,)
                traces[relation].y = traces[relation].y + (n_end,)
                traces[relation].z = traces[relation].z + (n_end - n_start,)

            # Add all traces to the figure
            for trace in traces.values():
                fig.add_trace(trace)

        """
        Add a time point to time point relationship
        
        Which means the z is 0, x = y, and only three relationships: before, equals, after
        """

        if self.mode == "timepoints":

            # define a center point for timepoint
            center_point = (50, 50)
            start, end = center_point
            fig.add_trace(
                go.Scatter3d(
                    x=[start],
                    y=[end],
                    z=[0],
                    mode="markers",
                    marker=dict(size=20, color="black", symbol="cross"),
                    showlegend=False,
                )
            )

            nearby_points = self.generate_nearby_timepoints(center_point)

            for nearby_point in nearby_points:
                n_start, n_end = nearby_point
                relation = self.determine_relation_timepoint(center_point, nearby_point)
                traces[relation].x = traces[relation].x + (n_start,)
                traces[relation].y = traces[relation].y + (n_end,)
                traces[relation].z = traces[relation].z + (0,)
            # Add all traces to the figure
            for trace in traces.values():
                fig.add_trace(trace)

        if self.mode == "timepointrange":
            center_point = (50, 50)
            start, end = center_point
            fig.add_trace(
                go.Scatter3d(
                    x=[start],
                    y=[end],
                    z=[0],
                    mode="markers",
                    marker=dict(size=20, color="black", symbol="cross"),
                    showlegend=False,
                )
            )

            nearby_points = self.generate_nearby_time_ranges(center_point)
            for nearby_point in nearby_points:
                n_start, n_end = nearby_point
                relation = self.determine_relation_timepointrange(
                    center_point, nearby_point
                )
                if relation is None:
                    continue
                traces[relation].x = traces[relation].x + (n_start,)
                traces[relation].y = traces[relation].y + (n_end,)
                traces[relation].z = traces[relation].z + (n_end - n_start,)
            # Add all traces to the figure
            for trace in traces.values():
                fig.add_trace(trace)

        if self.mode == "timerangepoint":
            center_point = (25, 75)
            start, end = center_point
            fig.add_trace(
                go.Scatter3d(
                    x=[start],
                    y=[end],
                    z=[end - start],
                    mode="markers",
                    marker=dict(size=20, color="black", symbol="cross"),
                    showlegend=False,
                )
            )

            nearby_points = self.generate_nearby_timepoints_for_timerange(center_point)
            for nearby_point in nearby_points:
                n_start, n_end = nearby_point
                relation = self.determine_relation_timerangepoint(
                    center_point, nearby_point
                )
                if relation is None:
                    continue
                traces[relation].x = traces[relation].x + (n_start,)
                traces[relation].y = traces[relation].y + (n_end,)
                traces[relation].z = traces[relation].z + (0,)
            # Add all traces to the figure
            for trace in traces.values():
                fig.add_trace(trace)

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title="Start Time",
                yaxis_title="End Time",
                zaxis_title="Total Time",
                xaxis=dict(range=[self.max_time, 0], autorange="reversed"),
                yaxis=dict(range=[self.max_time, 0], autorange="reversed"),
                zaxis=dict(range=[0, self.max_time * 2]),
                aspectmode="cube",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            title="Allen's 13 Temporal Relations Visualization",
            legend_title="Temporal Relations",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        # Show plot
        fig.show()
        # to html
        fig.write_html(f"allen_temporal_relations.html_{self.mode}")


if __name__ == "__main__":
    simulator = AllenTemporalSimulator(
        n=1, m=10000, max_time=100, mode="timerangepoint"
    )
    simulator.visualize()
