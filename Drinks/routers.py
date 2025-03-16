class CodeAnalysisRouter:
    """
    A router to control database operations for the CodeAnalysis model.
    """

    def db_for_read(self, model, **hints):
        """Point read operations for CodeAnalysis to the PostgreSQL DB."""
        if model._meta.app_label == 'code_analysis':  # ✅ Fix: Use correct app label
            return 'code_analysis'
        return None

    def db_for_write(self, model, **hints):
        """Point write operations for CodeAnalysis to the PostgreSQL DB."""
        if model._meta.app_label == 'code_analysis':  # ✅ Fix: Use correct app label
            return 'code_analysis'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """Allow relationships only within the same database."""
        if obj1._state.db == obj2._state.db:
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """Ensure that CodeAnalysis is migrated only to the 'code_analysis' database."""
        if app_label == 'code_analysis':  # ✅ Fix: Use correct app label
            return db == 'code_analysis'
        return None
