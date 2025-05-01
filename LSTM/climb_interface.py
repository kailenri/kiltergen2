from LSTM.climb_model import ClimbGenerator

class ClimbInterface:
    def __init__(self, model_path=None):
        self.generator = ClimbGenerator("climb_results.json")
        if model_path:
            self.generator.load_model(model_path)
        else:
            self.generator.train()

    def start_interactive(self):
        print("\nClimb Generation System")
        while True:
            print("\nOptions:")
            print("1. Generate from climb IDs")
            print("2. Generate from climb names")
            print("3. List available climbs")
            print("4. Exit")
            
            choice = input("Select an option: ").strip()
            
            if choice == "1":
                self._generate_by_ids()
            elif choice == "2":
                self._generate_by_names()
            elif choice == "3":
                self._list_climbs()
            elif choice == "4":
                break
            else:
                print("Invalid option")

    def _generate_by_ids(self):
        climb_ids = input("Enter climb IDs (comma separated): ").strip().split(',')
        climb_ids = [id.strip() for id in climb_ids]
        self.generator.generate(climb_ids)

    def _generate_by_names(self):
        name_query = input("Enter climb names (comma separated): ").strip().lower()
        climb_names = [name.strip() for name in name_query.split(',')]
        climb_ids = []
        
        for name in climb_names:
            found = False
            for climb_id, data in self.generator.climb_db.items():
                if name in data['name'].lower():
                    climb_ids.append(climb_id)
                    found = True
                    break
            if not found:
                print(f"No climb found matching: {name}")
        
        if climb_ids:
            self.generator.generate(climb_ids)

    def _list_climbs(self, limit=10):
        print("\nAvailable climbs (showing first 10):")
        for i, (climb_id, data) in enumerate(self.generator.climb_db.items()):
            if i >= limit:
                break
            print(f"{climb_id}: {data['name']}")

if __name__ == "__main__":
    interface = ClimbInterface(model_path="best_climb_lstm.pth")
    interface.start_interactive()