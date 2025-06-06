from argparse import ArgumentParser
from multiprocessing.dummy import Namespace
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import numpy as np


def load_data(path: str, separator: str = ';'):
	data = pd.read_csv(
		filepath_or_buffer=path,
		sep=separator
	)
	data = data.rename(columns={
		'Concepto': 'Concept',
		'Fecha': 'Date',
		'Importe': 'Amount',
		'Saldo disponible': 'Net Balance'
	})
	data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
	data['Amount'] = data['Amount']\
		.str.replace('EUR', '', regex=False)\
		.str.replace('.', '')\
		.str.replace(',', '.')
	data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')
	data = data.sort_values('Date')

	return data


def calculate_statistics(data):
	expenditures = data[data['Amount'] < 0]['Amount']
	average_daily_expenditure = expenditures.mean()
	average_monthly_expenditure = expenditures \
		.groupby(
			data['Date']
			.dt
			.to_period('M'))\
		.sum()\
		.mean()
	largest_purchase = expenditures.min()
	largest_purchase_date = None
	largest_purchase_concept = None

	if not data.loc[data['Amount'] == largest_purchase].empty:
		largest_purchase_date = data.loc[data['Amount']
										 == largest_purchase, 'Date'].iloc[0]
		largest_purchase_date = largest_purchase_date.strftime('%d/%m/%Y')
		largest_purchase_concept = data.loc[data['Amount']
											== largest_purchase, 'Concept'].iloc[0]

	print("📊 Statistics:")
	print(f"\t- Average daily expenditure: {average_daily_expenditure:.2f}")
	print(
		f"\t- Average monthly expenditure: {average_monthly_expenditure:.2f}")
	print(
		f"\t- Largest purchase: {largest_purchase:.2f} on {largest_purchase_date} for {largest_purchase_concept}")

def plot_net_balance(df, window_size=3):
	df['Day'] = df['Date'].dt.day
	df['Month'] = df['Date'].dt.to_period('M')

	df['Monthly Net Change'] = df.groupby('Month')['Amount'].cumsum()

	plt.figure(figsize=(12, 6))

	months = df['Month'].unique()

	for month in months:
		month_data = df[df['Month'] == month]

		# Aggregate by day (take max to remove duplicates)
		agg = month_data.groupby('Day')['Monthly Net Change'].max().reset_index()

		# Create full range of days for the month
		full_days = pd.Series(range(1, agg['Day'].max() + 1))

		# Reindex and forward fill missing days
		agg_full = agg.set_index('Day').reindex(full_days).ffill().reset_index()
		agg_full.rename(columns={'index': 'Day'}, inplace=True)

		x = agg_full['Day'].values
		y = agg_full['Monthly Net Change'].values

		# Apply moving average smoothing
		if len(y) >= window_size:
			y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
			x_smooth = x[(window_size - 1)//2 : -(window_size // 2)]
			plt.plot(x_smooth, y_smooth, label=str(month))
		else:
			plt.plot(x, y, marker='o', label=str(month))

	plt.title("Net Balance Change by Day of Month (Smoothed with Filled Days)")
	plt.xlabel("Day of Month")
	plt.ylabel("Net Balance Change (€)")
	plt.grid(True)
	plt.legend(title="Month")
	plt.tight_layout()
	plt.show()

def parse_args() -> Namespace:
	parser = ArgumentParser(
		prog='Finances App',
		description='A Python tool for analyzing personal bank income and expenditure data'
	)

	parser.add_argument('-i', '--input', type=str,
						help='Path to the CSV file with the transactions')

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input = None

	if args.input:
		input = args.input

	if not input:
		print('No input file')
		return

	data = load_data(input)
	print(data[data['Amount'] > 0]['Amount'])
	calculate_statistics(data)
	plot_net_balance(data)


if __name__ == "__main__":
	main()
