"""
Plotting functions for Gradient Descent notebook.
"""

import pandas as pd
import numpy as np
import altair as alt
from IPython.display import display, Markdown
import plotly.graph_objects as go
import plotly.express as px

class PlotManager:
    def __init__(self, show_chart=True, show_summary=True, show_table=True, 
                 high_contrast=False, base_font_size=14, table_points=10):
        self.show_chart = show_chart
        self.show_summary = show_summary
        self.show_table = show_table
        self.high_contrast = high_contrast
        self.base_font_size = base_font_size
        self.table_points = table_points

    def _apply_global_config(self, chart):
        """Standardizes the look of all charts as the final step."""
        return chart.configure(
            title=alt.TitleConfig(fontSize=self.base_font_size + 4, anchor='start'),
            axis=alt.AxisConfig(
                labelFontSize=self.base_font_size, 
                titleFontSize=self.base_font_size,
                gridWidth=1.5 if self.high_contrast else 1
            ),
            legend=alt.LegendConfig(labelFontSize=self.base_font_size, titleFontSize=self.base_font_size)
        ).configure_view(fill='white')

    def plot_function(self, x_range, func, title="Function Plot", header_level=2):
        df = pd.DataFrame({'x': x_range, 'y': func(x_range)})
        
        # Build raw chart
        line_color = "black" if self.high_contrast else "#1f77b4"
        chart = alt.Chart(df, title=title).mark_line(
            color=line_color, strokeWidth=4 if self.high_contrast else 2
        ).encode(
            x=alt.X('x', title='x'),
            y=alt.Y('y', title='f(x)'),
            tooltip=['x', 'y']
        ).properties(width=400, height=300, description=f"Line chart for {title}")

        if self.show_chart:
            # Configure at the very end
            display(self._apply_global_config(chart).interactive())
        
        if self.show_summary:
            self._display_summary(df, title, header_level)
        if self.show_table:
            self._display_table(df, header_level)

    def plot_gradient_descent(self, x_range, f, df_func, x_start, gamma, num_iter, title="Gradient Descent Steps", header_level=2):
        # 1. Data Prep
        curve_df = pd.DataFrame({'x': x_range, 'y': f(x_range)})
        steps = []
        curr_x = x_start
        for k in range(num_iter + 1):
            steps.append({'k': k, 'x': curr_x, 'y': f(curr_x)})
            if k < num_iter:
                curr_x -= gamma * df_func(curr_x)
        steps_df = pd.DataFrame(steps)

        # 2. Layer 1: The Function Curve
        line_color = "black" if self.high_contrast else "#1f77b4"
        base_curve = alt.Chart(curve_df).mark_line(
            color=line_color, strokeWidth=2, opacity=0.4 if not self.high_contrast else 0.6
        ).encode(x='x', y='y')

        # 3. Layer 2: The Descent Path (Points and Lines)
        path_color = "red" if not self.high_contrast else "#D55E00"
        path = alt.Chart(steps_df).mark_line(color=path_color, point=True, strokeWidth=3).encode(
            x='x', y='y', tooltip=['k', 'x', 'y']
        )
        
        # 4. Layer 3: Start/End Labels
        label_data = steps_df.iloc[[0, -1]] if num_iter > 0 else steps_df.iloc[[0]]
        labels = alt.Chart(label_data).mark_text(
            align='left', dx=10, fontSize=self.base_font_size, fontWeight='bold'
        ).encode(
            x='x', y='y',
            text=alt.condition(alt.datum.k == 0, alt.value('Start'), alt.value('End'))
        )

        # 5. Combine and THEN Configure
        layered_chart = alt.layer(base_curve, path, labels).properties(
            width=400, height=300, title=title,
            description=f"Gradient descent visualization for {title}"
        )

        if self.show_chart:
            display(self._apply_global_config(layered_chart).interactive())
        
        if self.show_summary:
            self._display_descent_summary(steps_df, gamma, title, header_level)
        if self.show_table:
            self._display_table(steps_df.rename(columns={'k': 'Iteration', 'y': 'f(x)'}), header_level)

    def _display_summary(self, df, title, header_level):
        hashes = "#" * header_level
        trend = "overall upward" if df['y'].iloc[-1] > df['y'].iloc[0] else "overall downward"
        
        summary_md = f"""
---
{hashes} Data Summary: {title}
* **Domain:** {df['x'].min():.1f} to {df['x'].max():.1f}
* **Range:** {df['y'].min():.2f} to {df['y'].max():.2f}
* **Trend:** The function exhibits an **{trend} trend**.
---
"""
        display(Markdown(summary_md))

    def _display_descent_summary(self, steps_df, gamma, title, header_level):
        hashes = "#" * header_level
        start_x, end_x = steps_df['x'].iloc[0], steps_df['x'].iloc[-1]
        
        summary_md = f"""
---
{hashes} Descent Narrative: {title}
* **Start:** x = {start_x:.4f} | **End:** x = {end_x:.4f}
* **Total Shift:** {end_x - start_x:.4f} in the x-dimension.
* **Parameters:** Learning rate ($\gamma$) = {gamma}.
---
"""
        display(Markdown(summary_md))

    def _display_table(self, df, header_level):
        hashes = "#" * header_level
        display(Markdown(f"{hashes} Select Data Samples"))
        step = max(1, len(df) // (self.table_points - 1))
        sampled = pd.concat([df.iloc[::step], df.iloc[-1:]]).drop_duplicates()
        style = sampled.style.hide(axis="index").format(precision=4)
        if self.high_contrast:
            style = style.set_table_styles([
                {'selector': 'th', 'props': [('border', '2px solid black'), ('color', 'black')]},
                {'selector': 'td', 'props': [('border', '1px solid black')]}
            ])
        display(style)

    def plot_gradient_descent_2d(self, x_range, y_range, f, df_func, start_point, gamma, num_iter, title="2D Gradient Descent"):
        """
        Visualizes gradient descent on a 2D surface using a contour plot.
        start_point: tuple (x, y)
        df_func: function returning tuple (dx, dy)
        """
        # 1. Prepare Surface Data for Contours
        X, Y = np.meshgrid(x_range, y_range)
        Z = f(X, Y)

        # 2. Run Gradient Descent 2D
        path = [(*start_point, f(*start_point))]
        curr_x, curr_y = start_point
        for _ in range(num_iter):
            dx, dy = df_func(curr_x, curr_y)
            curr_x -= gamma * dx
            curr_y -= gamma * dy
            path.append((curr_x, curr_y, f(curr_x, curr_y)))
        
        path_df = pd.DataFrame(path, columns=['x', 'y', 'z'])

        # 3. Create Plotly Figure
        fig = go.Figure()

        # Add Contours
        fig.add_trace(go.Contour(
            z=Z, x=x_range, y=y_range,
            colorscale='Viridis' if not self.high_contrast else 'Greys',
            contours=dict(showlabels=True),
            name='Function Contours',
            colorbar=dict(title="f(x,y)"),
            opacity=0.8
        ))

        # Add Descent Path
        path_color = "red" if not self.high_contrast else "cyan"
        fig.add_trace(go.Scatter(
            x=path_df['x'], y=path_df['y'],
            mode='lines+markers',
            marker=dict(size=8, color=path_color),
            line=dict(width=3, color=path_color),
            name='Descent Path'
        ))

        # Define label styling based on high contrast mode
        bg_color = "white" if not self.high_contrast else "black"
        font_color = "black" if not self.high_contrast else "white"
        border_color = "black" if not self.high_contrast else "white"

        # Add Start/End Annotations with high-visibility boxes
        for i, label in zip([0, -1], ["Start", "End"]):
            fig.add_annotation(
                x=path_df['x'].iloc[i], 
                y=path_df['y'].iloc[i],
                text=f"<b>{label}</b>",  # Bold text
                showarrow=True,
                arrowhead=2,
                arrowcolor=font_color,
                arrowsize=1.5,
                ax=20, ay=-30,          # Offsets the text so it doesn't cover the point
                font=dict(
                    size=self.base_font_size + 2,
                    color=font_color
                ),
                bgcolor=bg_color,       # Background color for the label box
                bordercolor=border_color,
                borderwidth=2,
                borderpad=4,
                opacity=0.9
            )

        # Apply Global Config Style
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="y",
            font=dict(size=self.base_font_size),
            width=600, height=600,
            xaxis=dict(
                # This forces the y-axis to match the scale of the x-axis
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                # Ensures the grid doesn't stretch
                constrain='domain'
            ),
            template="plotly_white"
        )

        if self.show_chart:
            fig.show()

        if self.show_summary:
            self._display_descent_summary_2d(path_df, gamma, title)

        if self.show_table:
            self._display_table(path_df, header_level=2)

    def _display_descent_summary_2d(self, path_df, gamma, title):
        start = path_df.iloc[0]
        end = path_df.iloc[-1]
        
        summary_md = f"""
---
### Descent Narrative: {title}
* **Starting Coordinates:** ({start.x:.4f}, {start.y:.4f}) with value **{start.z:.4f}**.
* **Final Coordinates:** ({end.x:.4f}, {end.y:.4f}) with value **{end.z:.4f}**.
* **Learning Rate ($\gamma$):** {gamma}
* **Path Observation:** The algorithm moved through **{len(path_df)-1}** steps of descent.
---
"""
        display(Markdown(summary_md))



    def plot_gradient_descent_3d(self, x_range, y_range, f, df_func, start_point, gamma, num_iter, title="3D Surface Descent"):
        """
        Visualizes gradient descent on a 3D surface plot.
        """
        # 1. Prepare Surface Data
        X, Y = np.meshgrid(x_range, y_range)
        Z = f(X, Y)

        # 2. Run Gradient Descent Pathing
        path = [(*start_point, f(*start_point))]
        curr_x, curr_y = start_point
        for _ in range(num_iter):
            dx, dy = df_func(curr_x, curr_y)
            curr_x -= gamma * dx
            curr_y -= gamma * dy
            path.append((curr_x, curr_y, f(curr_x, curr_y)))
        
        path_df = pd.DataFrame(path, columns=['x', 'y', 'z'])

        # 3. Create 3D Figure
        fig = go.Figure()

        # Add the Surface
        # Use 'Greys' for high contrast, 'Viridis' for standard
        colorscale = 'Greys' if self.high_contrast else 'Viridis'
        fig.add_trace(go.Surface(
            z=Z, x=x_range, y=y_range,
            colorscale=colorscale,
            opacity=0.8,
            showscale=False,
            name='Surface'
        ))

        # Add the Descent Path (Scatter3d)
        # We nudge the z-value up slightly (+0.1) so the line doesn't "sink" into the floor
        path_color = "red" if not self.high_contrast else "#D55E00"
        fig.add_trace(go.Scatter3d(
            x=path_df['x'], y=path_df['y'], z=path_df['z'] + (Z.max() * 0.01),
            mode='lines+markers',
            marker=dict(size=5, color=path_color, symbol='circle'),
            line=dict(width=6, color=path_color),
            name='Descent Path'
        ))

        # Update Layout for 3D Scene
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='f(X,Y)',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Sets a good initial view
            ),
            width=800, height=700,
            template="plotly_white",
            font=dict(size=self.base_font_size)
        )

        if self.show_chart:
            # Note: 3D plots require WebGL support in the browser
            display(fig)

        if self.show_summary:
            # We can reuse the 2D summary logic as the math is the same
            self._display_descent_summary_2d(path_df, gamma, f"{title} (3D View)")

        if self.show_table:
            self._display_table(path_df, header_level=2)