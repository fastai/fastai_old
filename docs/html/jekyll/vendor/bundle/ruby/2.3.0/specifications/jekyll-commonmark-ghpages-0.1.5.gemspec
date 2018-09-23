# -*- encoding: utf-8 -*-
# stub: jekyll-commonmark-ghpages 0.1.5 ruby lib

Gem::Specification.new do |s|
  s.name = "jekyll-commonmark-ghpages"
  s.version = "0.1.5"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["Ashe Connor"]
  s.date = "2018-02-12"
  s.email = "kivikakk@github.com"
  s.homepage = "https://github.com/github/jekyll-commonmark-ghpages"
  s.licenses = ["MIT"]
  s.rubygems_version = "2.5.2.1"
  s.summary = "CommonMark generator for Jekyll"

  s.installed_by_version = "2.5.2.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<jekyll-commonmark>, ["~> 1"])
      s.add_runtime_dependency(%q<commonmarker>, ["~> 0.17.6"])
      s.add_runtime_dependency(%q<rouge>, ["~> 2"])
      s.add_development_dependency(%q<rspec>, ["~> 3.0"])
      s.add_development_dependency(%q<rake>, [">= 0"])
      s.add_development_dependency(%q<bundler>, ["~> 1.6"])
    else
      s.add_dependency(%q<jekyll-commonmark>, ["~> 1"])
      s.add_dependency(%q<commonmarker>, ["~> 0.17.6"])
      s.add_dependency(%q<rouge>, ["~> 2"])
      s.add_dependency(%q<rspec>, ["~> 3.0"])
      s.add_dependency(%q<rake>, [">= 0"])
      s.add_dependency(%q<bundler>, ["~> 1.6"])
    end
  else
    s.add_dependency(%q<jekyll-commonmark>, ["~> 1"])
    s.add_dependency(%q<commonmarker>, ["~> 0.17.6"])
    s.add_dependency(%q<rouge>, ["~> 2"])
    s.add_dependency(%q<rspec>, ["~> 3.0"])
    s.add_dependency(%q<rake>, [">= 0"])
    s.add_dependency(%q<bundler>, ["~> 1.6"])
  end
end
