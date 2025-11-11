module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "type-enum": [
      2,
      "always",
      [
        "feat",     // new feature
        "fix",      // bug fix
        "docs",     // documentation only
        "style",    // code style (formatting, missing semi-colons, etc)
        "refactor", // refactoring code
        "perf",     // performance improvement
        "test",     // adding/modifying tests
        "chore",    // maintenance tasks
        "build",    // build system or dependencies
        "ci"        // CI/CD configuration changes
      ]
    ],
    "subject-case": [0, "never"]
  }
};

