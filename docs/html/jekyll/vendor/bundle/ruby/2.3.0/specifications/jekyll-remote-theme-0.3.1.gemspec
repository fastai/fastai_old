# -*- encoding: utf-8 -*-
# stub: jekyll-remote-theme 0.3.1 ruby lib

Gem::Specification.new do |s|
  s.name = "jekyll-remote-theme"
  s.version = "0.3.1"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["Ben Balter"]
  s.date = "2018-05-14"
  s.email = ["ben.balter@github.com"]
  s.homepage = "https://github.com/benbalter/jekyll-remote-theme"
  s.licenses = ["MIT"]
  s.rubygems_version = "2.5.2.1"
  s.summary = "Jekyll plugin for building Jekyll sites with any GitHub-hosted theme"

  s.installed_by_version = "2.5.2.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<jekyll>, ["~> 3.5"])
      s.add_runtime_dependency(%q<rubyzip>, ["< 3.0", ">= 1.2.1"])
      s.add_development_dependency(%q<jekyll-theme-primer>, ["~> 0.5"])
      s.add_development_dependency(%q<jekyll_test_plugin_malicious>, ["~> 0.2"])
      s.add_development_dependency(%q<pry>, ["~> 0.11"])
      s.add_development_dependency(%q<rspec>, ["~> 3.0"])
      s.add_development_dependency(%q<rubocop>, [">= 0.49.0", "~> 0.4"])
      s.add_development_dependency(%q<webmock>, ["~> 3.0"])
    else
      s.add_dependency(%q<jekyll>, ["~> 3.5"])
      s.add_dependency(%q<rubyzip>, ["< 3.0", ">= 1.2.1"])
      s.add_dependency(%q<jekyll-theme-primer>, ["~> 0.5"])
      s.add_dependency(%q<jekyll_test_plugin_malicious>, ["~> 0.2"])
      s.add_dependency(%q<pry>, ["~> 0.11"])
      s.add_dependency(%q<rspec>, ["~> 3.0"])
      s.add_dependency(%q<rubocop>, [">= 0.49.0", "~> 0.4"])
      s.add_dependency(%q<webmock>, ["~> 3.0"])
    end
  else
    s.add_dependency(%q<jekyll>, ["~> 3.5"])
    s.add_dependency(%q<rubyzip>, ["< 3.0", ">= 1.2.1"])
    s.add_dependency(%q<jekyll-theme-primer>, ["~> 0.5"])
    s.add_dependency(%q<jekyll_test_plugin_malicious>, ["~> 0.2"])
    s.add_dependency(%q<pry>, ["~> 0.11"])
    s.add_dependency(%q<rspec>, ["~> 3.0"])
    s.add_dependency(%q<rubocop>, [">= 0.49.0", "~> 0.4"])
    s.add_dependency(%q<webmock>, ["~> 3.0"])
  end
end
