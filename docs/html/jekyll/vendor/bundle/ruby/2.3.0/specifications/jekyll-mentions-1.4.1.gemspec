# -*- encoding: utf-8 -*-
# stub: jekyll-mentions 1.4.1 ruby lib

Gem::Specification.new do |s|
  s.name = "jekyll-mentions"
  s.version = "1.4.1"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["GitHub, Inc."]
  s.date = "2018-08-08"
  s.email = "support@github.com"
  s.homepage = "https://github.com/jekyll/jekyll-mentions"
  s.licenses = ["MIT"]
  s.required_ruby_version = Gem::Requirement.new(">= 2.3.0")
  s.rubygems_version = "2.5.2.1"
  s.summary = "@mention support for your Jekyll site"

  s.installed_by_version = "2.5.2.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<html-pipeline>, ["~> 2.3"])
      s.add_runtime_dependency(%q<jekyll>, ["~> 3.0"])
      s.add_development_dependency(%q<rake>, ["~> 12.0"])
      s.add_development_dependency(%q<rspec>, ["~> 3.0"])
      s.add_development_dependency(%q<rubocop>, ["~> 0.57.2"])
    else
      s.add_dependency(%q<html-pipeline>, ["~> 2.3"])
      s.add_dependency(%q<jekyll>, ["~> 3.0"])
      s.add_dependency(%q<rake>, ["~> 12.0"])
      s.add_dependency(%q<rspec>, ["~> 3.0"])
      s.add_dependency(%q<rubocop>, ["~> 0.57.2"])
    end
  else
    s.add_dependency(%q<html-pipeline>, ["~> 2.3"])
    s.add_dependency(%q<jekyll>, ["~> 3.0"])
    s.add_dependency(%q<rake>, ["~> 12.0"])
    s.add_dependency(%q<rspec>, ["~> 3.0"])
    s.add_dependency(%q<rubocop>, ["~> 0.57.2"])
  end
end
