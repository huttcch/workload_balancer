from flask import Flask, render_template, request, send_file, url_for, jsonify
import pandas as pd
import numpy as np
import random
import os
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # Not strictly needed if we process in memory
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Genetic Algorithm Implementation ---
class GeneticAlgorithm:
    def __init__(self, tasks_data, num_people, population_size=1000, generations=500):
        self.tasks_data = tasks_data
        self.num_people = num_people
        self.population_size = population_size
        self.generations = generations
        self.num_tasks = len(tasks_data)
        self.tasks_workload = np.array([task['workload'] for task in tasks_data])

    def _calculate_fitness(self, chromosome):
        """Calculates the variance of workload sums for each group."""
        group_workloads = [0] * self.num_people
        for i, label in enumerate(chromosome):
            group_workloads[label - 1] += self.tasks_workload[i]
        return np.var(group_workloads)

    def _generate_initial_chromosome(self):
        """Generates a random chromosome."""
        return [random.randint(1, self.num_people) for _ in range(self.num_tasks)]

    def _selection(self, population_with_fitness, num_to_select):
        """Selects chromosomes with the lowest fitness (variance)."""
        population_with_fitness.sort(key=lambda x: x[1])
        return [chrom for chrom, fitness in population_with_fitness[:num_to_select]]

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover."""
        if len(parent1) < 2: # Handle cases with very few tasks
            return parent1, parent2
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate_change(self, chromosome, mutation_rate=0.1):
        """Changes a random gene in the chromosome."""
        if random.random() < mutation_rate:
            idx = random.randint(0, len(chromosome) - 1)
            chromosome[idx] = random.randint(1, self.num_people)
        return chromosome
    
    def _mutate_swap(self, chromosome, mutation_rate=0.1):
        """Swaps two random genes in the chromosome."""
        if random.random() < mutation_rate and len(chromosome) >= 2:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def run(self):
        population = [self._generate_initial_chromosome() for _ in range(self.population_size)]
        best_chromosome = None
        best_fitness = float('inf')

        for generation in range(self.generations):
            population_with_fitness = [(chrom, self._calculate_fitness(chrom)) for chrom in population]
            
            # Update best chromosome found so far
            current_best_chrom, current_best_fit = min(population_with_fitness, key=lambda x: x[1])
            if current_best_fit < best_fitness:
                best_fitness = current_best_fit
                best_chromosome = current_best_chrom

            selected_parents = self._selection(population_with_fitness, 50) # Select 50 best parents

            next_population = []
            # Keep the best parents directly
            next_population.extend(selected_parents) 

            # Create new offspring through crossover and mutation
            while len(next_population) < self.population_size:
                if len(selected_parents) < 2: # Handle case if not enough parents for crossover
                    parent1 = random.choice(population) 
                    parent2 = random.choice(population)
                else:
                    parent1, parent2 = random.sample(selected_parents, 2)
                
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutate_change(child1, mutation_rate=0.2)
                child1 = self._mutate_swap(child1, mutation_rate=0.2)
                
                child2 = self._mutate_change(child2, mutation_rate=0.2)
                child2 = self._mutate_swap(child2, mutation_rate=0.2)

                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            population = next_population
            
            # Optional: print progress
            # if generation % 50 == 0:
            #     print(f"Generation {generation}, Best Fitness: {best_fitness}")

        # Ensure the final best_chromosome is truly from the last best
        final_population_with_fitness = [(chrom, self._calculate_fitness(chrom)) for chrom in population]
        final_best_chrom, final_best_fit = min(final_population_with_fitness, key=lambda x: x[1])
        if final_best_fit < best_fitness:
            best_fitness = final_best_fit
            best_chromosome = final_best_chrom


        return best_chromosome, best_fitness

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        try:
            df = pd.read_excel(file)
            
            if 'task' not in df.columns or 'workload' not in df.columns:
                return jsonify({'error': 'Excel file must contain "task" and "workload" columns.'}), 400

            tasks_data = df[['task', 'workload']].to_dict(orient='records')
            
            num_people = int(request.form.get('num_people', 1))
            if num_people <= 0:
                return jsonify({'error': 'Number of people must be a positive integer.'}), 400
            
            if len(tasks_data) < num_people:
                return jsonify({'error': 'Number of tasks must be greater than or equal to the number of people.'}), 400

            # Run Genetic Algorithm
            ga = GeneticAlgorithm(tasks_data, num_people)
            best_labels, final_variance = ga.run()

            # Create output DataFrame
            output_df = df.copy()
            output_df['label'] = best_labels

            # Prepare file for download
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            output_df.to_excel(writer, index=False, sheet_name='Workload Distribution')
            writer.close()
            output.seek(0)

            # Generate a temporary file path for download link, or directly send it
            # For direct download:
            # return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
            #                 as_attachment=True, download_name='workload_distribution.xlsx')

            # For a link:
            # Store the data in session or temporary storage if needed, but for simplicity, 
            # we'll just return a success message and trust the client to handle the next step.
            # A more robust solution for multiple users might store the generated file temporarily
            # and provide a unique URL. For this example, we'll indicate success.
            
            # You can send the output_df back as JSON if you want to display it on the page
            # Or, we can just return a success and trigger download from client.
            # Let's try sending a base64 encoded version for direct download.
            
            # If you want to return the output_df to display on screen:
            # output_data_for_display = output_df.to_dict(orient='split')
            # return jsonify({
            #     'success': True, 
            #     'message': 'Workload calculated successfully!',
            #     'results': output_data_for_display,
            #     'variance': final_variance
            # })

            # For simplicity, we'll just indicate success and let the client know it's ready for download
            return jsonify({
                'success': True,
                'message': 'Workload calculated successfully!',
                'variance': final_variance,
                'download_url': url_for('download_results') # This is a placeholder for actual download.
                                                            # We'll handle sending the file in the download_results route.
            })

        except Exception as e:
            app.logger.error(f"Error processing file: {e}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload an Excel file (.xlsx or .xls).'}), 400

@app.route('/download_results', methods=['GET'])
def download_results():
    # This route will only work if the data is saved in memory or a temp file
    # For a simple single-user app, we can store the last generated df in a global variable
    # In a production multi-user app, you would need a more robust way to store per-user files.
    
    # For demonstration, let's assume `output_df` is stored in `g` or a global variable
    # A better approach would be to pass a unique ID and retrieve the file.
    if hasattr(app, 'last_output_df') and app.last_output_df is not None:
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        app.last_output_df.to_excel(writer, index=False, sheet_name='Workload Distribution')
        writer.close()
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                         as_attachment=True, download_name='workload_distribution.xlsx')
    else:
        return jsonify({'error': 'No results to download. Please upload a file first.'}), 404

if __name__ == '__main__':
    app.run(debug=True)