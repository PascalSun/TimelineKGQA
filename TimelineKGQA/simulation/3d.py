import numpy as np
import plotly.graph_objects as go


class AllenTemporalSimulator:
    def __init__(self, n, m, max_time=10):
        self.n = n  # number of random points to generate
        self.m = m  # number of points to generate around each random point
        self.max_time = max_time  # maximum time value for all axes
        self.color_shape_map = {
            "before": ("red", "circle"),
            "after": ("red", "square"),
            "meets": ("blue", "circle"),
            "met_by": ("blue", "square"),
            "overlaps": ("green", "circle"),
            "overlapped_by": ("green", "square"),
            "starts": ("purple", "circle"),
            "started_by": ("purple", "square"),
            "during": ("orange", "circle"),
            "contains": ("orange", "square"),
            "finishes": ("yellow", "circle"),
            "finished_by": ("yellow", "square"),
            "equals": ("cyan", "diamond"),
        }

    def generate_surface(self):
        x = np.linspace(0, self.max_time, 100)
        y = np.linspace(0, self.max_time, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.maximum(Y - X, 0)  # Z starts from 0 at the origin and increases
        # remove all points where X > Y
        Z[X > Y] = np.nan
        return X, Y, Z

    def generate_points(self):
        points = []
        for _ in range(self.n):
            start = np.random.uniform(0, self.max_time * 0.8)
            end = np.random.uniform(start + 0.1, self.max_time)
            points.append((start, end))
        return points

    def generate_nearby_points(self, point):
        start, end = point
        nearby_points = []
        for _ in range(self.m):
            new_start = np.random.uniform(
                max(0, start - 100), min(self.max_time, start + 100)
            )
            new_end = np.random.uniform(
                max(new_start + 0.1, end - 100), min(self.max_time, end + 100)
            )
            nearby_points.append((new_start, new_end))
        return nearby_points

    def determine_relation(self, interval1, interval2):
        start1, end1 = interval1
        start2, end2 = interval2

        if end1 < start2:
            return "before"
        elif start1 > end2:
            return "after"
        elif end1 == start2:
            return "meets"
        elif start1 == end2:
            return "met_by"
        elif start1 < start2 < end1 < end2:
            return "overlaps"
        elif start2 < start1 < end2 < end1:
            return "overlapped_by"
        elif start1 == start2 and end1 < end2:
            return "starts"
        elif start1 == start2 and end1 > end2:
            return "started_by"
        elif start2 < start1 < end1 < end2:
            return "during"
        elif start1 < start2 < end2 < end1:
            return "contains"
        elif start1 < start2 < end2 == end1:
            return "finishes"
        elif start2 < start1 < end1 == end2:
            return "finished_by"
        elif start1 == start2 and end1 == end2:
            return "equals"

    def visualize(self):
        X, Y, Z = self.generate_surface()

        fig = go.Figure()

        # Add reference surface
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                showscale=False,
                opacity=0.3,
            )
        )

        # Generate random points
        points = self.generate_points()

        # Create a trace for each relation
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

        # Visualize points and their relations
        for point in points:
            start, end = point
            fig.add_trace(
                go.Scatter3d(
                    x=[start],
                    y=[end],
                    z=[end - start],
                    mode="markers",
                    marker=dict(size=7, color="black", symbol="cross"),
                    showlegend=False,
                )
            )

            nearby_points = self.generate_nearby_points(point)
            for nearby_point in nearby_points:
                n_start, n_end = nearby_point
                relation = self.determine_relation(point, nearby_point)
                traces[relation].x = traces[relation].x + (n_start,)
                traces[relation].y = traces[relation].y + (n_end,)
                traces[relation].z = traces[relation].z + (n_end - n_start,)

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


if __name__ == "__main__":
    simulator = AllenTemporalSimulator(1, 1000000, max_time=100)
    simulator.visualize()
