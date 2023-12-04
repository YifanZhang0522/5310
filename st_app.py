import streamlit as st
from streamlit_option_menu import option_menu

import pickle
from pathlib import Path
import streamlit_authenticator as stauth


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
import base64




st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# -------- user authentication ----------

names = ['Emily Johnson', 'Alex Smith', 'Laura Miller', 'Yifan Zhang']
usernames = ['EmilyJ_88', 'AlexS_45', 'LauraM_23', 'YifanZ_22']

# load hashed passwords
file_path = Path(__file__).parent / 'hashed_pw.pk1'
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)
    
authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'dashboard', 'abcdef', cookie_expiry_days = 30)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == False:
    st.error('Username/password is incorret')

if authentication_status == None:
    st.error('Please enter your username and password')



# -------- dashboard main contents ----------
if authentication_status:
   
    # sidebar
    
    
    
    authenticator.logout('logout', 'sidebar')
    
    # logo
    logo_path = "/Users/zhangshuxin/Desktop/AA/APAN 5310/project/logo2.jpeg"  # Replace with your image path
    st.sidebar.image(logo_path, use_column_width=True)

    st.sidebar.markdown(f'## Welcome {name}')
    
    st.sidebar.header('ABC Food Mart Database Dashboard `version 1`')


    # option menu

    with st.sidebar:
        selected = option_menu(
            menu_title = 'Main Menu',
            options = ['Overview', 
                    'Direct Query',
                    'Query Viewer', 
                    'Total Sales & Costs by Store',
                    'Customer Segment',
                    'Customer Media Usage',
                    'Customer Counts by Promotion'],
        )


    if selected == 'Overview':
        st.title('Database Overview')
        # -----intro content------
        
        # ------connect to database-------
        # Pass the connection string to a variable, conn_url
        conn_url = 'postgresql://postgres:123@localhost/5310_Project'

        # Create an engine that connects to PostgreSQL server
        engine = create_engine(conn_url)

        # Establish a connection
        connection = engine.connect()
        
        # customer count
        sql_query = """
        select count(*)
        from customer
        """
    
        customer_count_df = pd.read_sql(sql_query, engine)
        
        # store count
        sql_query = """
        select count(*)
        from store
        """
    
        store_count_df = pd.read_sql(sql_query, engine)
        
        # food count
        sql_query = """
        select count(*)
        from food_info
        """
    
        food_count_df = pd.read_sql(sql_query, engine)
        
    
        # key metrics
        col1, col2, col3 = st.columns(3)
        total_customer = customer_count_df['count'].iloc[0]
        total_store = store_count_df['count'].iloc[0]
        total_food = food_count_df['count'].iloc[0]
        col1.metric('Number of Customers', total_customer)
        col2.metric('Number of Stores', total_store)
        col3.metric('Number of Food', total_food)
        
        
        
        
        # sales & costs key metrics
        sql_query = """
        select avg(store_sales) as avg_sales, avg(store_cost) as avg_costs
        from sales
        """
    
        avg_sales_costs_df = pd.read_sql(sql_query, engine)
        
        col4, col5 = st.columns(2)
        avg_sales_num = round(avg_sales_costs_df['avg_sales'].iloc[0],2)
        avg_costs_num = round(avg_sales_costs_df['avg_costs'].iloc[0],2)
        avg_sales = f'{avg_sales_num}M'
        avg_costs = f'{avg_costs_num}M'
        col4.metric('Average Store Sales', avg_sales)
        col5.metric('Average Store Costs', avg_costs)
    
    # ----- store map -----
        st.markdown('### Store Locations')
        
        # from geopy.geocoders import Nominatim
        # import folium
        # from streamlit_folium import folium_static
        
        # # query data from database
        # sql_query = """
        # select distinct store_id, store_type, store_city, store_state, store_sqft, sales_country
        # from store
        # left join sales
        # using (store_id)
        # """
        
        # # save as dataframe
        # store_location_df = pd.read_sql(sql_query, engine)

        # # Aggregate data
        # city_counts = store_location_df.groupby(['store_city', 'store_state']).size().reset_index(name='count')

        # # Define default map center (center of the US)
        # default_latitude = 39.8283
        # default_longitude = -98.5795
        
        # # Initialize map
        # m = folium.Map(location=[default_latitude, default_longitude], zoom_start=3)

        # # Geocode and add markers
        # geolocator = Nominatim(user_agent="streamlit_app", timeout = 10)
        # for _, row in city_counts.iterrows():
        #     location = geolocator.geocode(f"{row['store_city']}, {row['store_state']}")
        #     if location:
        #         folium.CircleMarker(
        #             [location.latitude, location.longitude],
        #             radius=row['count'] * 1.5,  
        #             popup=f"{row['store_city']}: {row['count']} stores",
        #             fill=True
        #         ).add_to(m)

        # # Display map 
        # folium_static(m)
            
        
    # ----- Direct Query ------
    
    if selected == 'Direct Query':
        # Pass the connection string to a variable, conn_url
        # note here we use the readonly user identity to avoid changing the database
        conn_url = 'postgresql://readonly_user:secure_password@localhost/5310_Project'

        # Create an engine that connects to PostgreSQL server
        engine = create_engine(conn_url)

        # Establish a connection
        connection = engine.connect()
            
        # sql query executor 
        def sql_executor(raw_code):
            try: 
                result = connection.execute(raw_code)
                return pd.DataFrame(result.fetchall(), columns = result.keys())
            except SQLAlchemyError as e:
                return str(e)

        st.title("Direct Query: Read-Only")

        # enter sql code window
        with st.form(key='query_form'):
            raw_code = st.text_area("SQL Code Here")
            submit_code = st.form_submit_button("Execute")

            
        # Results Layouts
        if submit_code:
            st.info("Query Submitted")
            st.code(raw_code)

            # Results 
            query_results = sql_executor(raw_code)
            
            if isinstance(query_results, pd.DataFrame):
                with st.expander("Results"):
                    query_df = pd.DataFrame(query_results)
                    st.dataframe(query_df)
            else: 
                st.error(f"Error executing query: {query_results}, Only SELECT queries are allowed")

        
        
    # ----- Query Viewer -----------    
    
    if selected == 'Query Viewer':
        # intro content  
        
        
        # Pass the connection string to a variable, conn_url
        conn_url = 'postgresql://postgres:123@localhost/5310_Project'

        # Create an engine that connects to PostgreSQL server
        engine = create_engine(conn_url)

        # Establish a connection
        connection = engine.connect()

        # Function to execute a SQL query and return the result as a Pandas DataFrame
        def execute_query(query):
            result = connection.execute(text(query))
            result_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return result_df

        # Streamlit app
        st.title("Query Viewer")

        # Query buttons
        query_buttons = {
            "Store Information": "SELECT * FROM store ORDER BY store_id;",
            "The Most Popular Items of All Stores (The Number of People Purchasing Each Food)": """
                SELECT fi.food_id, brand_name, srp AS "Suggested Retail Price", count(customer_id) AS Number_of_people_purchasing 
                FROM customer_food cf
                INNER JOIN food_info fi
                ON cf.food_id=fi.food_id
                GROUP BY fi.food_id, brand_name, srp 
                ORDER BY count(customer_id) desc;
            """,
            "Top 10 Most Popular Brands of Each Store": """
                select store_id, brand_name, brand_popularity
                from (
                    select 
                        store_id, 
                        brand_name, 
                        count(*) as brand_popularity,
                        row_number() over (partition by store_id order by count(*) desc) as rn
                    from food_store
                    left join food_info 
                    using (food_id)
                    group by store_id, brand_name
                ) as ranked_brands
                where rn <= 10
                order by store_id, brand_popularity desc;
            """,
            "Customer Profile (Income Level)": """
                SELECT
                    avg_yearly_income,
                    COUNT(customer_id) AS Number_of_customers,
                    CONCAT(ROUND((COUNT(customer_id) * 100.0 / SUM(COUNT(customer_id)) OVER()), 2), '%') AS Percentage
                FROM
                    customer
                GROUP BY
                    avg_yearly_income
                ORDER BY
                    CASE
                        WHEN avg_yearly_income = '$10K - $30K' THEN 1
                        WHEN avg_yearly_income = '$30K - $50K' THEN 2
                        WHEN avg_yearly_income = '$50K - $70K' THEN 3
                        WHEN avg_yearly_income = '$70K - $90K' THEN 4
                        WHEN avg_yearly_income = '$90K - $110K' THEN 5
                        WHEN avg_yearly_income = '$110K - $130K' THEN 6
                        WHEN avg_yearly_income = '$130K - $150K' THEN 7
                        WHEN avg_yearly_income = '$150K +' THEN 8
                    END;
            """,
                        "Main Customer Profile of Each Brand": """
                WITH brand_summary AS (
                    SELECT fi.brand_name,
                        COUNT(*) AS total_customers,
                        COUNT(*) FILTER (WHERE c.gender = 'M') AS male_count,
                        COUNT(*) FILTER (WHERE c.gender = 'F') AS female_count,
                        COUNT(*) FILTER (WHERE c.education = 'Graduate Degree') AS grad_degree_count,
                        COUNT(*) FILTER (WHERE c.education = 'Partial College') AS partial_college_count,
                        COUNT(*) FILTER (WHERE c.education = 'High School Degree') AS high_school_count,
                        COUNT(*) FILTER (WHERE c.education = 'Bachelors Degree') AS bachelors_count,
                        COUNT(*) FILTER (WHERE c.education = 'Partial High School') AS partial_high_school_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$150K +') AS income_150k_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$70K - $90K') AS income_70k_90k_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$30K - $50K') AS income_30k_50k_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$50K - $70K') AS income_50k_70k_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$10K - $30K') AS income_10k_30k_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$130K - $150K') AS income_130k_150k_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$90K - $110K') AS income_90k_110k_count,
                        COUNT(*) FILTER (WHERE c.avg_yearly_income = '$110K - $130K') AS income_110k_130k_count,
                        ROW_NUMBER() OVER (PARTITION BY fi.brand_name ORDER BY COUNT(*) DESC) AS rn
                    FROM customer c
                    JOIN customer_food cf ON c.customer_id = cf.customer_id
                    JOIN food_info fi ON cf.food_id = fi.food_id
                    GROUP BY fi.brand_name
                )
                SELECT brand_name,
                    CASE WHEN male_count > female_count THEN 'M' ELSE 'F' END AS main_customer_gender,
                    (
                        SELECT category
                        FROM (VALUES 
                            ('Graduate Degree', grad_degree_count),
                            ('Partial College', partial_college_count),
                            ('High School Degree', high_school_count),
                            ('Bachelors Degree', bachelors_count),
                            ('Partial High School', partial_high_school_count)
                        ) AS edu(category, count)
                        ORDER BY count DESC
                        LIMIT 1
                    ) AS main_customer_education,
                    (
                        SELECT income
                        FROM (VALUES 
                            ('$150K +', income_150k_count),
                            ('$70K - $90K', income_70k_90k_count),
                            ('$30K - $50K', income_30k_50k_count),
                            ('$50K - $70K', income_50k_70k_count),
                            ('$10K - $30K', income_10k_30k_count),
                            ('$130K - $150K', income_130k_150k_count),
                            ('$90K - $110K', income_90k_110k_count),
                            ('$110K - $130K', income_110k_130k_count)
                        ) AS inc(income, count)
                        ORDER BY count DESC
                        LIMIT 1
                    ) AS main_customer_avg_yearly_income
                FROM brand_summary
                WHERE rn = 1;
            """,
            "Popular Media Types": """
                SELECT media_type, 
                COUNT(DISTINCT customer_id) AS customer_count 
                FROM customer_media 
                GROUP BY media_type
                ORDER BY customer_count DESC;
            """,
            "Top 10 Most Popular Promotion & Media Combination of Each Store": """
                select store_id, media_type, promotion_name, customer_count
                from (
                    select 
                        store_id,
                        media_type,
                        promotion_name,
                        count(customer_id) as customer_count,
                        row_number() over (partition by store_id order by count(*) desc) as rn
                    from customer_media
                    inner join customer_promo
                    using (customer_id)
                    inner join promotion
                    using (promotion_id)
                    inner join customer_store
                    using (customer_id)
                    group by store_id, media_type, promotion_name
                ) as ranked_media_promo
                where rn <= 10
                order by store_id, customer_count desc;
            """
            

            
            # Add more queries as needed
        }

        # Display query buttons
        selected_query = st.selectbox("Select a query:", list(query_buttons.keys()))

        # Execute the selected query
        result_df = execute_query(query_buttons[selected_query])
        
        # Display the result as a table
        st.dataframe(result_df, height = 300)
        
        # Show the SQL query used
        with st.expander("Show SQL Query"):
            st.code(query_buttons[selected_query], language='sql')

       
        # Function to create a download link for a DataFrame
        def get_table_download_link(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download CSV</a>'
            return href

         # Download button
        st.markdown(get_table_download_link(result_df), unsafe_allow_html=True)

        

    if selected == 'Total Sales & Costs by Store':
        st.title('Total Sales & Costs by Store')
        # store_sales bar chart
        # ----- connect to database --------
   
        # Pass the connection string to a variable, conn_url
        conn_url = 'postgresql://postgres:123@localhost/5310_Project'

        # Create an engine that connects to PostgreSQL server
        engine = create_engine(conn_url)

        # Establish a connection
        connection = engine.connect()
        
        # ----- Total Sales -------
        # Query to get data for store sales
        query_sales = 'SELECT store_id, SUM(store_sales) as total_sales FROM sales GROUP BY store_id;'
        df_sales = pd.read_sql(query_sales, engine)
        
        # first graph: store sales
        df_sales['store_id'] = df_sales['store_id'].astype(str)

        # Drop rows with missing values
        df_sales.dropna(subset=['store_id', 'total_sales'], inplace=True)

        # Display the DataFrame in Streamlit
        # st.write("DataFrame Preview:", df_sales.head())
        
        
        # Create a Plotly bar chart
        fig_sales = px.bar(df_sales, x='store_id', y='total_sales',
             labels={'store_id': 'Store ID', 'total_sales': 'Total Sales'},
             title='Total Sales by Store')


        
        
        # ----- Total Costs -------
        # Query to get data for store sales
        query_costs = 'SELECT store_id, SUM(store_cost) as total_costs FROM sales GROUP BY store_id;'
        df_costs = pd.read_sql(query_costs, engine)
        
        # first graph: store sales
        df_costs['store_id'] = df_costs['store_id'].astype(str)

        # Drop rows with missing values
        df_costs.dropna(subset=['store_id', 'total_costs'], inplace=True)

        # Display the DataFrame in Streamlit
        # st.write("DataFrame Preview:", df_sales.head())
        
        
        # Create a Plotly bar chart
        fig_costs = px.bar(df_costs, x='store_id', y='total_costs',
             labels={'store_id': 'Store ID', 'total_costs': 'Total Costs'},
             title='Total Costs by Store')


        
        
        # ----- side by side plot -------
        df_combined = pd.merge(df_sales, df_costs, on='store_id', how='inner')

        # Create a long format DataFrame for Plotly
        df_long = pd.melt(df_combined, id_vars='store_id', value_vars=['total_sales', 'total_costs'])

        # Create a Plotly bar chart with grouped bars
        fig_cb = px.bar(df_long, x='store_id', y='value', color='variable',
                    labels={'store_id': 'Store ID', 'value': 'Amount', 'variable': 'Type'},
                    title='Total Sales and Costs by Store')

        # Modify the layout for better readability
        fig_cb.update_layout(barmode='group')

        # Display the Plotly bar chart in Streamlit
        st.plotly_chart(fig_cb)

        # Display the Plotly bar chart in Streamlit
        # st.plotly_chart(fig_sales)
        
        # Display the Plotly bar chart in Streamlit
        # st.plotly_chart(fig_costs)
        
        
        # # Creating the plot
        # fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
        # ax.bar(df_sales['store_id'], df_sales['total_sales'])

        # # Adding labels and title for clarity
        # ax.set_xlabel('Store ID')
        # ax.set_ylabel('Total Sales')
        # ax.set_title('Total Sales by Store')
        # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # # Display the plot in Streamlit
        # st.pyplot(fig)
        
    if selected == 'Customer Segment':
        
        st.title('Customer Segment')
        
        # Pass the connection string to a variable, conn_url
        conn_url = 'postgresql://postgres:123@localhost/5310_Project'

        # Create an engine that connects to PostgreSQL server
        engine = create_engine(conn_url)

        # Establish a connection
        connection = engine.connect()
        
        # ----- Customer Segment of All Stores (Income Level) ------
        # extract data from database
        query_customer_segment_all = '''
        SELECT
            avg_yearly_income,
            COUNT(customer_id) AS Number_of_customers
            
        FROM
            customer
        GROUP BY
            avg_yearly_income
        ORDER BY
            CASE
                WHEN avg_yearly_income = '$10K - $30K' THEN 1
                WHEN avg_yearly_income = '$30K - $50K' THEN 2
                WHEN avg_yearly_income = '$50K - $70K' THEN 3
                WHEN avg_yearly_income = '$70K - $90K' THEN 4
                WHEN avg_yearly_income = '$90K - $110K' THEN 5
                WHEN avg_yearly_income = '$110K - $130K' THEN 6
                WHEN avg_yearly_income = '$130K - $150K' THEN 7
                WHEN avg_yearly_income = '$150K +' THEN 8 
            END;
        '''
        df_customer_segment_all = pd.read_sql(query_customer_segment_all, engine)
        
        # Calculate the total number of customers
        total_customers = df_customer_segment_all['number_of_customers'].sum()

        # Calculate percentage
        df_customer_segment_all['percentage'] = df_customer_segment_all['number_of_customers']/ total_customers

        # Create a pie chart
        fig = px.pie(df_customer_segment_all, names='avg_yearly_income', values='percentage', 
                    title='Percentage Distribution of Customers by Income Level', 
                    hole=0.3)  

        # Update layout for better readability
        # fig.update_traces(textposition='inside', textinfo='percent+label')

        # Display the figure
        st.plotly_chart(fig)
        
        # Query for Customer Segment by Store
        query_customer_segment = '''
        SELECT cs.store_id, c.avg_yearly_income, COUNT(*) as count 
        FROM customer c
        JOIN customer_store cs ON c.customer_id = cs.customer_id
        GROUP BY cs.store_id, c.avg_yearly_income;
        '''
        df_customer_segment = pd.read_sql(query_customer_segment, engine)

        
        # pivot_df = df_customer_segment.pivot(index='store_id', columns='avg_yearly_income', values='count').fillna(0)
        # fig2, ax = plt.subplots()
        # pivot_df.plot(kind='bar', stacked=True, ax=ax)
        # st.pyplot(fig2)
        
        # plotly 
        # fig = go.Figure()

        # # Add a bar trace for each column in the pivot table
        # for col in pivot_df.columns:
        #     fig.add_trace(go.Bar(
        #         x=pivot_df.index,
        #         y=pivot_df[col],
        #         name=col
        #     ))

        # # Update layout for a stacked bar chart
        # fig.update_layout(
        #     barmode='stack',
        #     title='',
        #     xaxis_title='Store ID',
        #     yaxis_title='Count',
        #     legend_title='Average Yearly Income'
        # )
        
        # st.plotly_chart(fig)
        
        # ----- percentage stack bar chart ------
        # Calculate percentage
        df_customer_segment['percentage'] = df_customer_segment.groupby('store_id')['count'].apply(lambda x: x / x.sum())

        # Pivot the DataFrame
        pivot_df = df_customer_segment.pivot(index='store_id', columns='avg_yearly_income', values='percentage').fillna(0)

        # Create an empty figure
        fig = go.Figure()

        # Add a bar trace for each income level
        for col in pivot_df.columns:
            fig.add_trace(go.Bar(
                x=pivot_df.index,
                y=pivot_df[col],
                name=str(col)  # column name as the label
            ))

        # Update layout for a stacked bar chart
        fig.update_layout(
            barmode='stack',
            title='Percentage of Customers by Income Level in Each Store',
            xaxis_title='Store ID',
            yaxis_title='Percentage (%)',
            yaxis=dict(tickformat=".0%"),  # Format y-axis ticks as percentages
            legend_title='Average Yearly Income'
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)
        
        
        
        
        
        
    if selected == 'Customer Media Usage':
        st.title('Customer Media Usage')
        # customer_media bar chart
        # ----- connect to database --------
   
        # Pass the connection string to a variable, conn_url
        conn_url = 'postgresql://postgres:123@localhost/5310_Project'

        # Create an engine that connects to PostgreSQL server
        engine = create_engine(conn_url)

        # Establish a connection
        connection = engine.connect()
        
        sql_query = """
        select media_type, count(media_type) as number_of_customer
        from customer_media
        group by media_type
        order by count(media_type) desc
        """
        
        df = pd.read_sql(sql_query, engine)
        
        fig = px.bar(df, x='media_type', y='number_of_customer',
             labels={'media_type': 'Media Type', 'number_of_customer': 'Number of Customers'},
             title='Number of Customers Using Each Type of Media')


        # Display the Plotly bar chart in Streamlit
        st.plotly_chart(fig)
        
    if selected == 'Customer Counts by Promotion':
        # Pass the connection string to a variable, conn_url
        conn_url = 'postgresql://postgres:123@localhost/5310_Project'

        # Create an engine that connects to PostgreSQL server
        engine = create_engine(conn_url)

        # Establish a connection
        connection = engine.connect()
        
        # Query for Count per Promotion Name
        query_promotion = '''
        SELECT p.promotion_name, COUNT(DISTINCT cp.customer_id) as count 
        FROM customer_promo cp
        JOIN promotion p ON cp.promotion_id = p.promotion_id
        GROUP BY p.promotion_name
        ORDER BY COUNT(DISTINCT cp.customer_id) desc;
        '''
        df_promotion = pd.read_sql(query_promotion, engine)

        st.title('Customer Counts by Promotion')
        # fig4, ax = plt.subplots()
        # ax.bar(df_promotion['promotion_name'], df_promotion['count'])
        # st.pyplot(fig4) 
        
        # Create a Plotly bar chart
        fig = px.bar(df_promotion, x='promotion_name', y='count',
             labels={'promotion_name': 'Promotion Name', 'count': 'Count'},
             title='Count by Promotion Name')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
            
        
        
        
        
        




   
